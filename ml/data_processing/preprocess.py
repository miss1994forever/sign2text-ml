#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing module for the Sign2Text ML pipeline.

This module contains functions for preprocessing sign language video data,
including loading, cleaning, normalizing, and preparing data for model training.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import List, Dict, Tuple, Optional, Union, Any
import mediapipe as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


class SignVideoProcessor:
    """Process sign language videos for feature extraction."""

    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 use_landmarks: bool = True,
                 extract_hands: bool = True,
                 extract_pose: bool = True,
                 extract_face: bool = False,
                 normalize: bool = True):
        """
        Initialize the video processor.

        Args:
            target_size: Target frame size as (width, height)
            use_landmarks: Whether to extract pose landmarks
            extract_hands: Whether to extract hand landmarks
            extract_pose: Whether to extract body pose landmarks
            extract_face: Whether to extract facial landmarks
            normalize: Whether to normalize landmarks
        """
        self.target_size = target_size
        self.use_landmarks = use_landmarks
        self.extract_hands = extract_hands
        self.extract_pose = extract_pose
        self.extract_face = extract_face
        self.normalize = normalize

        # Initialize MediaPipe components
        if self.use_landmarks:
            self.holistic = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )

    def process_video(self, video_path: str, max_frames: int = 60) -> Dict[str, Any]:
        """
        Process a video file and extract features.

        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process

        Returns:
            Dictionary containing processed features
        """
        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        # Determine frame sampling
        if frame_count > max_frames:
            sample_interval = frame_count / max_frames
        else:
            sample_interval = 1

        # Initialize containers
        frames = []
        hand_landmarks = []
        pose_landmarks = []
        face_landmarks = []

        frame_idx = 0
        next_sample = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= next_sample:
                # Resize frame
                frame = cv2.resize(frame, self.target_size)

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract landmarks if enabled
                if self.use_landmarks:
                    results = self.holistic.process(rgb_frame)

                    # Extract hand landmarks
                    if self.extract_hands:
                        left_hand = self._extract_landmarks(results.left_hand_landmarks)
                        right_hand = self._extract_landmarks(results.right_hand_landmarks)
                        hand_landmarks.append({
                            'left': left_hand,
                            'right': right_hand
                        })

                    # Extract pose landmarks
                    if self.extract_pose:
                        pose = self._extract_landmarks(results.pose_landmarks)
                        pose_landmarks.append(pose)

                    # Extract face landmarks
                    if self.extract_face:
                        face = self._extract_landmarks(results.face_landmarks)
                        face_landmarks.append(face)

                # Add processed frame
                frames.append(frame)
                next_sample += sample_interval

            frame_idx += 1

        cap.release()

        # Create result dictionary
        result = {
            'frames': np.array(frames),
            'metadata': {
                'original_fps': fps,
                'original_frame_count': frame_count,
                'duration': duration,
                'processed_frame_count': len(frames)
            }
        }

        if self.use_landmarks:
            if self.extract_hands:
                result['hand_landmarks'] = hand_landmarks
            if self.extract_pose:
                result['pose_landmarks'] = pose_landmarks
            if self.extract_face:
                result['face_landmarks'] = face_landmarks

        return result

    def _extract_landmarks(self, landmark_list) -> Optional[np.ndarray]:
        """
        Extract landmarks from MediaPipe result.

        Args:
            landmark_list: MediaPipe landmark list

        Returns:
            Numpy array of landmark coordinates or None if no landmarks detected
        """
        if landmark_list is None:
            return None

        landmarks = []
        for landmark in landmark_list.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        landmarks = np.array(landmarks)

        # Normalize landmarks
        if self.normalize and landmarks.size > 0:
            landmarks = self._normalize_landmarks(landmarks)

        return landmarks

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be invariant to scale and translation.

        Args:
            landmarks: Array of landmark coordinates

        Returns:
            Normalized landmark coordinates
        """
        # Center the landmarks
        centroid = np.mean(landmarks, axis=0)
        centered = landmarks - centroid

        # Scale to unit std
        scale = np.std(centered)
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered

        return normalized


def process_dataset(video_dir: str,
                   annotation_file: Optional[str] = None,
                   output_dir: str = None,
                   max_frames: int = 60) -> pd.DataFrame:
    """
    Process all videos in a directory and create a dataset.

    Args:
        video_dir: Directory containing video files
        annotation_file: Path to annotation file (csv or json)
        output_dir: Directory to save processed features
        max_frames: Maximum number of frames per video

    Returns:
        DataFrame with processed data information
    """
    logger.info(f"Processing dataset in {video_dir}")

    # Load annotations if available
    annotations = None
    if annotation_file and os.path.exists(annotation_file):
        if annotation_file.endswith('.csv'):
            annotations = pd.read_csv(annotation_file)
        elif annotation_file.endswith('.json'):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize processor
    processor = SignVideoProcessor()

    # Get list of video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mpg', '.mpeg']:
        video_files.extend(list(Path(video_dir).glob(f'**/*{ext}')))

    results = []

    # Process each video
    for video_path in video_files:
        try:
            video_name = video_path.stem
            logger.info(f"Processing {video_name}")

            # Get annotation for this video if available
            label = None
            if annotations is not None:
                if isinstance(annotations, pd.DataFrame):
                    video_anno = annotations[annotations['filename'] == video_name]
                    if not video_anno.empty:
                        label = video_anno.iloc[0].get('label')
                elif isinstance(annotations, dict):
                    label = annotations.get(video_name, {}).get('label')

            # Process the video
            features = processor.process_video(str(video_path), max_frames=max_frames)

            # Save features if output directory specified
            if output_dir:
                feature_path = os.path.join(output_dir, f"{video_name}.npz")
                np.savez_compressed(
                    feature_path,
                    frames=features['frames'],
                    hand_landmarks=features.get('hand_landmarks', None),
                    pose_landmarks=features.get('pose_landmarks', None),
                    face_landmarks=features.get('face_landmarks', None),
                    metadata=features['metadata']
                )

            # Add to results
            results.append({
                'filename': video_name,
                'label': label,
                'frame_count': len(features['frames']),
                'duration': features['metadata']['duration'],
                'feature_path': feature_path if output_dir else None
            })

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")

    # Create and return summary dataframe
    return pd.DataFrame(results)


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8,
                 val_ratio: float = 0.1, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        df: DataFrame containing dataset information
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing train, val, and test DataFrames
    """
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split indices
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)

    # Split the dataset
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess sign language videos")
    parser.add_argument("--video_dir", required=True, help="Directory containing video files")
    parser.add_argument("--annotation_file", help="Path to annotation file (csv or json)")
    parser.add_argument("--output_dir", help="Directory to save processed features")
    parser.add_argument("--max_frames", type=int, default=60, help="Maximum number of frames per video")

    args = parser.parse_args()

    # Process the dataset
    dataset_df = process_dataset(
        args.video_dir,
        args.annotation_file,
        args.output_dir,
        args.max_frames
    )

    # Split the dataset
    splits = split_dataset(dataset_df)

    # Save split information
    if args.output_dir:
        for split_name, split_df in splits.items():
            split_df.to_csv(os.path.join(args.output_dir, f"{split_name}_set.csv"), index=False)

        # Save full dataset info
        dataset_df.to_csv(os.path.join(args.output_dir, "dataset_info.csv"), index=False)

    logger.info(f"Preprocessing complete. Total samples: {len(dataset_df)}")
    logger.info(f"Train set: {len(splits['train'])}, Validation set: {len(splits['val'])}, Test set: {len(splits['test'])}")
