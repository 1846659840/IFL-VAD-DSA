import os
import cv2
import json
import random
import numpy as np
import torch
import torchvision.transforms as transforms

def load_video_data(partition_id, num_partitions, dataset_path="/content/XD-Violence"):
    """Load video data from XD-Violence dataset"""
    # Set random seed for reproducibility
    random.seed(partition_id)
    np.random.seed(partition_id)

    # XD-Violence dataset paths
    video_dir = os.path.join(dataset_path, "videos")
    annotation_file = os.path.join(dataset_path, "annotations.json")

    # Check if paths exist
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")

    # Load annotation file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Get list of video files
    video_files = []
    for file in os.listdir(video_dir):
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_name = os.path.splitext(file)[0]
            if video_name in annotations:
                video_files.append(file)

    # If video files are insufficient, try other possible video formats
    if len(video_files) == 0:
        print("No standard format video files found, trying to find all possible video files")
        video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
        print(f"Files found: {video_files[:5]}...")

    # Ensure we have enough videos
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {video_dir}")

    print(f"Found {len(video_files)} video files in total")

    # Load test split list to exclude test videos
    test_split_file = os.path.join(dataset_path, "test_split.json")
    if os.path.exists(test_split_file):
        with open(test_split_file, 'r') as f:
            test_split = json.load(f)
        
        # Filter out test videos
        video_files = [file for file in video_files if os.path.splitext(file)[0] not in test_split]
        print(f"After excluding test videos: {len(video_files)} videos remaining")
    else:
        print("Test split file not found, using all available videos")

    # Assign videos to current client
    total_videos = len(video_files)
    videos_per_client = max(1, total_videos // num_partitions)

    start_idx = partition_id * videos_per_client
    end_idx = min((partition_id + 1) * videos_per_client, total_videos)

    # If this is the last client, ensure all remaining videos are included
    if partition_id == num_partitions - 1:
        end_idx = total_videos

    client_video_files = video_files[start_idx:end_idx]

    # Limit the number of videos per client for performance testing
    max_videos_per_client = 10
    if len(client_video_files) > max_videos_per_client:
        client_video_files = random.sample(client_video_files, max_videos_per_client)

    print(f"Client {partition_id} assigned {len(client_video_files)} videos")

    # Load videos assigned to this client
    videos = []
    successful_videos = 0

    for video_file in client_video_files:
        video_path = os.path.join(video_dir, video_file)
        video_segments = load_single_video(video_path)
        
        if video_segments:
            videos.append(video_segments)
            successful_videos += 1
            print(f"Successfully loaded video {video_file}, extracted {len(video_segments)} segments")

    # If no videos were successfully loaded, raise exception
    if successful_videos == 0:
        raise ValueError(f"Client {partition_id} failed to load any videos, please check dataset path and video format")

    print(f"Client {partition_id} successfully loaded {successful_videos} videos")

    return videos

def load_single_video(video_path):
    """Process a single video and extract frame segments
    
    Args:
        video_path: Path to the video file
        
    Returns:
        List of tensor segments, each containing 16 frames
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to standard input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Unable to open video: {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select key frames for large videos, process 16 frames per segment
        frames_per_segment = 16

        # Calculate sampling interval, ensure at least 4 segments
        min_segments = 4
        max_segments = 10

        # Calculate start frame for each segment
        if frame_count <= frames_per_segment * min_segments:
            # Short video, sample all frames
            segment_starts = list(range(0, frame_count, frames_per_segment))
        else:
            # Long video, uniform sampling
            num_segments = min(max_segments, frame_count // frames_per_segment)
            segment_stride = frame_count // num_segments
            segment_starts = [i * segment_stride for i in range(num_segments)]

        # Read frames for each segment
        video_segments = []

        for start_frame in segment_starts:
            # Set read position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read frames for one segment
            segment_frames = []
            for _ in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break  # Video ended

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Apply transformations
                frame_tensor = transform(frame)
                segment_frames.append(frame_tensor)

            # If segment contains enough frames, add to segment list
            if len(segment_frames) == frames_per_segment:
                # Stack all frames in the segment as tensor [frames_per_segment, channels, height, width]
                segment_tensor = torch.stack(segment_frames, dim=0)
                video_segments.append(segment_tensor)

        # Release video resource
        cap.release()
        
        return video_segments if video_segments else None

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

def load_test_data(dataset_path="/content/XD-Violence"):
    """Load test data with ground truth labels from XD-Violence dataset
    
    Args:
        dataset_path: Path to the XD-Violence dataset
        
    Returns:
        test_videos: List of video segments for testing
        test_labels: Binary labels (0: normal, 1: violent) for each video
    """
    video_dir = os.path.join(dataset_path, "videos")
    annotation_file = os.path.join(dataset_path, "annotations.json")
    test_split_file = os.path.join(dataset_path, "test_split.json")
    
    # Check if paths exist
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")
    if not os.path.exists(test_split_file):
        raise FileNotFoundError(f"Test split file does not exist: {test_split_file}")
    
    # Load test split list and annotations
    with open(test_split_file, 'r') as f:
        test_split = json.load(f)
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Load only test videos
    test_videos = []
    test_labels = []
    
    print(f"Loading {len(test_split)} test videos...")
    successful_videos = 0
    
    for video_name in test_split:
        # Try different extensions
        for ext in ['.mp4', '.avi', '.mov']:
            video_path = os.path.join(video_dir, video_name + ext)
            if os.path.exists(video_path):
                # Load video segments
                video_segments = load_single_video(video_path)
                if video_segments:
                    test_videos.append(video_segments)
                    
                    # Get ground truth label for the video
                    # Process annotation format: convert to binary label where any violence is labeled as 1
                    if isinstance(annotations[video_name], bool):
                        # Direct boolean annotation
                        label = 1 if annotations[video_name] else 0
                    elif isinstance(annotations[video_name], list):
                        # Temporal annotations - if any segment is violent, label the video as violent
                        label = 1 if any(annotations[video_name]) else 0
                    elif isinstance(annotations[video_name], dict):
                        # Dictionary format with labels
                        label = 1 if annotations[video_name].get("violence", False) else 0
                    else:
                        # Default to 0 if format is unknown
                        label = 0
                    
                    test_labels.append(label)
                    successful_videos += 1
                    print(f"Successfully loaded test video {video_name}, label: {label}")
                
                # Video found and processed (or failed), break the extension loop
                break
    
    print(f"Successfully loaded {successful_videos} test videos out of {len(test_split)}")
    
    return test_videos, test_labels 