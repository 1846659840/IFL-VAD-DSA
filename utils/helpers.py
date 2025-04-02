import numpy as np
import torch
from collections import OrderedDict
from typing import List
from sklearn.metrics import roc_auc_score

def get_parameters(net) -> List[np.ndarray]:
    """Get model parameters as a list of numpy arrays"""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    """Set model parameters"""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def extract_features(video, feature_extractor, device):
    """Extract features from video sequence"""
    features = []
    for frame in video:
        frame_tensor = frame.to(device)
        with torch.no_grad():
            feature = feature_extractor(frame_tensor.unsqueeze(0)).squeeze(0)
        features.append(feature.cpu().numpy())
    return np.array(features)  # Returns shape [seq_len, feature_dim] feature sequence

def compute_statistical_representation(features):
    """Compute statistical representation"""
    m_i = len(features)
    if m_i <= 2:
        return [0.0, 0.0]  # Handle edge cases

    # Calculate norms
    norms = [np.linalg.norm(feat, ord=2) for feat in features]

    # Mean of norm differences
    norm_diffs = [abs(norms[j] - norms[j+1]) for j in range(m_i-1)]
    mu_i = np.mean(norm_diffs)

    # Standard deviation of norm differences
    sigma_i = np.std(norm_diffs, ddof=1)

    # Average feature norm
    f_bar_i = np.mean(norms)

    # Norm variability
    G_i = 0
    for j in range(m_i):
        for k in range(m_i):
            G_i += abs(norms[j] - norms[k])
    G_i /= (2 * (m_i**2) * f_bar_i) if f_bar_i > 0 else 1

    return [sigma_i, G_i]

def coarse_grained_pseudo_label_assignment(statistical_representations, percentile=80):
    """Coarse-grained pseudo-label assignment"""
    # Calculate local center
    center = np.mean(statistical_representations, axis=0)

    # Calculate distances
    distances = [np.linalg.norm(x - center) for x in statistical_representations]

    # Set radius as p-percentile
    radius = np.percentile(distances, percentile)

    # Assign pseudo-labels
    pseudo_labels = [1 if d > radius else 0 for d in distances]

    return pseudo_labels

def fine_grained_pseudo_label_assignment(anomalous_indices, all_features, pca_matrix, normal_features, window_length=5, anomaly_threshold=3.0):
    """Assign fine-grained pseudo-labels for anomalous video segments"""
    fine_grained_labels = {}

    # Flatten and project normal features
    all_normal_segments = []
    for video_feats in normal_features:
        all_normal_segments.extend(video_feats)

    projected_normal = [np.dot(pca_matrix.T, feat) for feat in all_normal_segments]

    # Calculate mean and covariance
    mu = np.mean(projected_normal, axis=0)
    sigma_diag = np.diag(np.var(projected_normal, axis=0))
    sigma_inv = np.linalg.inv(sigma_diag)

    for idx in anomalous_indices:
        video_features = all_features[idx]
        m_i = len(video_features)

        # Project features
        projected_features = [np.dot(pca_matrix.T, feat) for feat in video_features]

        # Calculate Mahalanobis distance
        distances = []
        for z_ij in projected_features:
            diff = z_ij - mu
            d_ij = np.sqrt(np.dot(np.dot(diff, sigma_inv), diff))
            distances.append(d_ij)

        # Calculate cumulative sum
        cum_sum = np.cumsum([0] + distances)

        # Find window with maximum average
        max_avg = 0
        max_t = 0

        for t in range(1, m_i - window_length + 2):
            window_avg = (cum_sum[t + window_length - 1] - cum_sum[t - 1]) / window_length
            if window_avg > max_avg:
                max_avg = window_avg
                max_t = t - 1

        # Assign fine-grained labels
        segment_labels = []
        for j in range(m_i):
            if j in range(max_t, max_t + window_length) or distances[j] > anomaly_threshold:
                segment_labels.append(1)  # Violence
            else:
                segment_labels.append(0)  # Non-violence

        fine_grained_labels[idx] = segment_labels

    return fine_grained_labels

def evaluate_on_test_set(model, feature_extractor, test_videos, test_labels, device):
    """Evaluate model on test set with ground truth labels using AUC metric
    
    Args:
        model: The violence detection model
        feature_extractor: Feature extraction model
        test_videos: List of video segments for testing
        test_labels: Ground truth labels (0: normal, 1: violent)
        device: Device to run inference on (CPU or GPU)
        
    Returns:
        auc: Area Under the ROC Curve score
        all_preds: Predicted probabilities for each video
        all_true_labels: Ground truth labels
    """
    print("Evaluating model on test set...")
    model.eval()
    all_preds = []
    all_true_labels = []
    
    for video_segments, true_label in zip(test_videos, test_labels):
        # Extract features for each segment
        segment_features = []
        for segment in video_segments:
            segment_tensor = segment.to(device)
            with torch.no_grad():
                features = feature_extractor(segment_tensor.unsqueeze(0))
                segment_features.append(features.squeeze(0))
        
        # Process each segment with the detector model
        segment_preds = []
        for features in segment_features:
            features = features.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(features)
                pred_prob = torch.sigmoid(output).item()
                segment_preds.append(pred_prob)
        
        # Video-level prediction: max probability across segments
        # Use max for anomaly detection (if any segment is violent, the video is violent)
        video_pred = max(segment_preds) if segment_preds else 0.5
        
        all_preds.append(video_pred)
        all_true_labels.append(true_label)
    
    # Calculate AUC score (handling edge case with only one class)
    if len(set(all_true_labels)) > 1:
        auc = roc_auc_score(all_true_labels, all_preds)
    else:
        print("Warning: Only one class present in test labels, AUC is undefined")
        auc = 0.5  # Default value when AUC is undefined
    
    print(f"Test set evaluation - AUC: {auc:.4f}")
    
    return auc, all_preds, all_true_labels

def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataloader using AUC metric
    
    Args:
        model: The violence detection model
        dataloader: DataLoader containing evaluation data
        device: Device to run inference on
        
    Returns:
        auc: Area Under the ROC Curve score
        loss: Average loss value
    """
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.float().to(device).view(-1, 1)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Collect predictions and labels
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            true_labels = labels.cpu().numpy().flatten()
            
            all_preds.extend(probs)
            all_labels.extend(true_labels)
            
            # Accumulate loss
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    
    # Calculate metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    # Calculate AUC (handling edge case with only one class)
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        print("Warning: Only one class present in labels, AUC is undefined")
        auc = 0.5  # Default value when AUC is undefined
    
    return auc, avg_loss 