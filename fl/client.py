import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from flwr.client import NumPyClient
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

from ..utils.helpers import (
    get_parameters, 
    set_parameters, 
    extract_features,
    compute_statistical_representation,
    coarse_grained_pseudo_label_assignment,
    fine_grained_pseudo_label_assignment
)
from ..data.dataset import VideoSequenceDataset, create_data_loaders

class IFLVADDSAClient(NumPyClient):
    def __init__(self, partition_id, videos, feature_extractor, model, pca_matrix,
                 window_length=5, anomaly_threshold=3.0, percentile=80, device="cpu"):
        self.partition_id = partition_id
        self.videos = videos
        self.feature_extractor = feature_extractor
        self.model = model
        self.pca_matrix = pca_matrix
        self.window_length = window_length
        self.anomaly_threshold = anomaly_threshold
        self.percentile = percentile
        self.current_round = 0  # Track current round
        self.device = device

        # Generate pseudo-labels (Phase I)
        self.features, self.pseudo_labels, self.statistical_info = self.generate_pseudo_labels()

        # Create dataset and data loaders
        self.trainloader, self.valloader = self.create_data_loaders()

    def generate_pseudo_labels(self):
        """Implement Phase I: Local Pseudo-Label Generation"""
        print(f"[Client {self.partition_id}] Generating pseudo-labels")

        # Extract features for all videos
        all_features = [extract_features(video, self.feature_extractor, self.device) for video in self.videos]

        # Compute statistical representation
        statistical_reps = [compute_statistical_representation(features) for features in all_features]

        # Coarse-grained pseudo-label assignment
        coarse_labels = coarse_grained_pseudo_label_assignment(statistical_reps, self.percentile)

        # Identify normal and anomalous videos
        normal_indices = [i for i, label in enumerate(coarse_labels) if label == 0]
        anomalous_indices = [i for i, label in enumerate(coarse_labels) if label == 1]

        normal_features = [all_features[i] for i in normal_indices]

        # Assign fine-grained pseudo-labels for anomalous videos
        fine_grained_labels = {}
        if normal_features:  # Only when we have normal videos as reference
            fine_grained_labels = fine_grained_pseudo_label_assignment(
                anomalous_indices, all_features, self.pca_matrix, normal_features,
                self.window_length, self.anomaly_threshold
            )

        # Combine coarse-grained and fine-grained labels
        pseudo_labels = {}
        for i, label in enumerate(coarse_labels):
            if i in anomalous_indices and i in fine_grained_labels:
                pseudo_labels[i] = (label, fine_grained_labels[i])
            else:
                pseudo_labels[i] = (label, None)

        # Calculate auxiliary statistical information
        avg_sigma = np.mean([sr[0] for sr in statistical_reps])
        avg_g = np.mean([sr[1] for sr in statistical_reps])

        return all_features, pseudo_labels, (avg_sigma, avg_g)

    def create_data_loaders(self):
        """Create data loaders based on pseudo-labels"""
        train_sequences = []
        train_labels = []

        # Create dataset based on pseudo-labels
        for i, video in enumerate(self.videos):
            # Extract features
            feature_sequence = torch.tensor(self.features[i], dtype=torch.float32)
            coarse_label, fine_labels = self.pseudo_labels[i]

            if fine_labels is None:
                # For normal videos, use video-level label
                train_sequences.append(feature_sequence)
                train_labels.append(float(coarse_label))
            else:
                # For anomalous videos, use segment-level labels
                # Create sliding windows, each with a label
                for j in range(len(fine_labels) - self.window_length + 1):
                    window_features = feature_sequence[j:j+self.window_length]
                    window_label = 1 if sum(fine_labels[j:j+self.window_length]) > 0 else 0
                    train_sequences.append(window_features)
                    train_labels.append(float(window_label))

        # Create training and validation data loaders
        return create_data_loaders(train_sequences, train_labels, batch_size=16)

    def get_parameters(self, config):
        """Get model parameters"""
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Implement Phase II: Local Model Training and Update"""
        print(f"[Client {self.partition_id}] fit, config: {config}")

        # Get round information from config
        self.current_round = config.get("server_round", 1)

        # Update threshold (if provided by server)
        if "anomaly_threshold" in config:
            self.anomaly_threshold = config["anomaly_threshold"]
            print(f"[Client {self.partition_id}] Updated anomaly threshold to: {self.anomaly_threshold}")

        # Set model parameters
        set_parameters(self.model, parameters)

        # Get learning rate from config
        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 1)

        # Train model using binary cross-entropy loss
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for sequences, labels in self.trainloader:
                sequences = sequences.to(self.device)
                labels = labels.float().to(self.device).view(-1, 1)

                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[Client {self.partition_id}] Epoch {epoch+1}: loss {total_loss/len(self.trainloader)}")

        # Calculate local update
        updated_parameters = get_parameters(self.model)

        # Evaluate model quality - using AUC
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for sequences, labels in self.valloader:
                sequences = sequences.to(self.device)
                labels = labels.float().to(self.device)
                outputs = self.model(sequences)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        print(f"[Client {self.partition_id}] Model AUC: {auc}")

        # If quality is low or server forces update, update pseudo-labels
        label_update_stats = None
        if auc < 0.6 or config.get("force_label_update", False):
            label_update_stats = self.update_pseudo_labels()

        # Return parameters, number of examples, and metrics
        return updated_parameters, len(self.trainloader.dataset), {
            "auc": auc,
            "avg_sigma": self.statistical_info[0],
            "avg_g": self.statistical_info[1],
            "label_updates": label_update_stats
        }

    def update_pseudo_labels(self):
        """Update pseudo-labels based on model predictions - enhanced version"""
        self.model.eval()

        # Save original pseudo-labels for comparison
        original_pseudo_labels = self.pseudo_labels.copy()

        # Regenerate predictions and pseudo-labels for all videos
        new_pseudo_labels = {}
        update_statistics = {"changed_video_labels": 0, "changed_segment_labels": 0}

        for i, video in enumerate(self.videos):
            # Extract feature sequence
            feature_sequence = torch.tensor(self.features[i], dtype=torch.float32).to(self.device)

            # Extract original labels
            orig_video_label, orig_segment_labels = original_pseudo_labels.get(i, (0, None))

            # Apply model to make predictions
            segment_probs = []
            segment_preds = []
            for j in range(len(feature_sequence) - self.window_length + 1):
                window = feature_sequence[j:j+self.window_length].unsqueeze(0)
                with torch.no_grad():
                    output = self.model(window)
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob > 0.5 else 0
                    segment_probs.append(prob)
                    segment_preds.append(pred)

            # Determine video-level label - using probability average rather than just binary prediction
            avg_prob = sum(segment_probs) / len(segment_probs) if segment_probs else 0.5
            model_video_label = 1 if avg_prob > 0.5 else 0

            # Blend original labels and model predictions - confidence weighted
            confidence_weight = min(0.2 * self.current_round, 0.8)  # Increase model weight as training progresses
            final_video_label = round(model_video_label * confidence_weight +
                                     orig_video_label * (1 - confidence_weight))

            # If predicted as violence, assign segment-level labels
            if final_video_label == 1:
                # Assign labels for each segment of the video
                full_segment_labels = []

                # Process all segments
                for j in range(len(feature_sequence)):
                    # Find all windows containing this segment
                    window_predictions = []
                    window_probs = []
                    for w in range(max(0, j - self.window_length + 1), min(j + 1, len(segment_preds))):
                        window_predictions.append(segment_preds[w])
                        window_probs.append(segment_probs[w])

                    # If no windows contain this segment, use nearest window
                    if not window_predictions:
                        nearest_window = min(max(0, j - self.window_length + 1),
                                            len(segment_preds) - 1 if segment_preds else 0)
                        pred = segment_preds[nearest_window] if segment_preds else 0
                        orig_pred = orig_segment_labels[j] if orig_segment_labels and j < len(orig_segment_labels) else 0
                        # Blend original label and model prediction
                        final_pred = round(pred * confidence_weight + orig_pred * (1 - confidence_weight))
                    else:
                        # Calculate weighted prediction for this segment across all windows
                        weighted_pred = sum(window_probs) / len(window_probs) if window_probs else 0.5
                        model_pred = 1 if weighted_pred > 0.5 else 0

                        # Get original label
                        orig_pred = orig_segment_labels[j] if orig_segment_labels and j < len(orig_segment_labels) else 0

                        # Blend original label and model prediction
                        final_pred = round(model_pred * confidence_weight + orig_pred * (1 - confidence_weight))

                    full_segment_labels.append(final_pred)

                # Track label changes
                if orig_video_label != final_video_label:
                    update_statistics["changed_video_labels"] += 1

                if orig_segment_labels:
                    changes = sum(1 for a, b in zip(orig_segment_labels, full_segment_labels) if a != b)
                    update_statistics["changed_segment_labels"] += changes

                new_pseudo_labels[i] = (final_video_label, full_segment_labels)
            else:
                if orig_video_label != final_video_label:
                    update_statistics["changed_video_labels"] += 1

                new_pseudo_labels[i] = (final_video_label, None)

        # Update pseudo-labels
        self.pseudo_labels = new_pseudo_labels

        # Recreate data loaders
        self.trainloader, self.valloader = self.create_data_loaders()

        print(f"[Client {self.partition_id}] Label update statistics: Changed video labels: {update_statistics['changed_video_labels']}, "
              f"Changed segment labels: {update_statistics['changed_segment_labels']}")

        # Return label update statistics for server use
        return update_statistics

    def evaluate(self, parameters, config):
        """Evaluate model, using AUC as metric"""
        print(f"[Client {self.partition_id}] evaluate, config: {config}")

        # If updated PCA matrix is provided, update it
        if "pca_matrix" in config:
            self.pca_matrix = np.array(config["pca_matrix"])

        # If updated anomaly threshold is provided, update it
        if "anomaly_threshold" in config:
            self.anomaly_threshold = config["anomaly_threshold"]

        # Set model parameters
        set_parameters(self.model, parameters)

        # Evaluate model
        criterion = torch.nn.BCEWithLogitsLoss()
        self.model.eval()
        loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sequences, labels in self.valloader:
                sequences = sequences.to(self.device)
                labels = labels.float().to(self.device).view(-1, 1)
                outputs = self.model(sequences)
                loss += criterion(outputs, labels).item() * labels.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())

        # Calculate AUC
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        avg_loss = loss / len(self.valloader.dataset) if len(self.valloader.dataset) > 0 else 0

        return avg_loss, len(self.valloader.dataset), {"auc": auc} 