#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trainer module for the Sign2Text ML pipeline.

This module provides training functionality for sign language recognition models.
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.cnn_lstm import create_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignVideoDataset(Dataset):
    """Dataset for loading sign language video data."""

    def __init__(
        self,
        data_csv: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        use_landmarks: bool = True,
        max_seq_len: int = 60
    ):
        """
        Initialize the dataset.

        Args:
            data_csv: Path to the CSV file with annotations
            root_dir: Directory containing the data files
            transform: Optional transform to apply to frames
            use_landmarks: Whether to load landmark data
            max_seq_len: Maximum sequence length
        """
        self.data_df = pd.read_csv(data_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.use_landmarks = use_landmarks
        self.max_seq_len = max_seq_len

        # Create label to index mapping
        self.classes = sorted(self.data_df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        logger.info(f"Initialized dataset with {len(self.data_df)} samples and {len(self.classes)} classes")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing data for the requested sample
        """
        sample_info = self.data_df.iloc[idx]

        # Load preprocessed data
        feature_path = os.path.join(self.root_dir, sample_info['feature_path'])
        data = np.load(feature_path, allow_pickle=True)

        # Get frames and pad/truncate if needed
        frames = data['frames']
        seq_len = min(frames.shape[0], self.max_seq_len)

        # Create padded array
        padded_frames = np.zeros((self.max_seq_len, *frames.shape[1:]), dtype=frames.dtype)
        padded_frames[:seq_len] = frames[:seq_len]

        # Convert to tensor and permute to (seq_len, channels, height, width)
        frames_tensor = torch.FloatTensor(padded_frames).permute(0, 3, 1, 2)

        # Apply transform if available
        if self.transform:
            # Apply transform to each frame
            transformed_frames = []
            for i in range(seq_len):
                transformed_frames.append(self.transform(frames_tensor[i]))

            # Pad with zeros
            for i in range(seq_len, self.max_seq_len):
                transformed_frames.append(torch.zeros_like(transformed_frames[0]))

            frames_tensor = torch.stack(transformed_frames)

        # Load landmarks if needed
        landmarks_tensor = None
        if self.use_landmarks and 'hand_landmarks' in data:
            hand_landmarks = data['hand_landmarks']

            # Process and pad landmarks
            landmarks_list = []
            for i in range(min(len(hand_landmarks), self.max_seq_len)):
                # Extract left and right hand landmarks
                left = hand_landmarks[i]['left']
                right = hand_landmarks[i]['right']

                # Flatten and concatenate
                if left is not None and right is not None:
                    # Both hands visible
                    combined = np.concatenate([left.flatten(), right.flatten()])
                elif left is not None:
                    # Only left hand visible
                    combined = np.concatenate([left.flatten(), np.zeros_like(left.flatten())])
                elif right is not None:
                    # Only right hand visible
                    combined = np.concatenate([np.zeros(63), right.flatten()])
                else:
                    # No hands visible
                    combined = np.zeros(126)

                landmarks_list.append(combined)

            # Pad if needed
            while len(landmarks_list) < self.max_seq_len:
                landmarks_list.append(np.zeros(126))

            landmarks_tensor = torch.FloatTensor(np.array(landmarks_list))

        # Get label
        label = self.class_to_idx[sample_info['label']]
        label_tensor = torch.LongTensor([label])[0]

        # Create output dictionary
        output = {
            'frames': frames_tensor,
            'label': label_tensor,
            'seq_len': torch.LongTensor([seq_len])[0]
        }

        if landmarks_tensor is not None:
            output['landmarks'] = landmarks_tensor

        return output


class Trainer:
    """Trainer class for sign language recognition models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        output_dir: str = './output',
        num_epochs: int = 100,
        early_stopping: int = 10
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use for training
            output_dir: Directory to save outputs
            num_epochs: Number of epochs to train
            early_stopping: Number of epochs to wait for improvement before stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        logger.info(f"Trainer initialized with {len(train_loader.dataset)} training samples")
        logger.info(f"Using device: {device}")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch in pbar:
                # Move batch to device
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                # Handle different model input types
                if 'landmarks' in batch and hasattr(self.model, 'forward') and 'landmarks' in self.model.forward.__code__.co_varnames:
                    landmarks = batch['landmarks'].to(self.device)
                    outputs = self.model(frames, landmarks)
                else:
                    outputs = self.model(frames)

                # Get logits
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                # Calculate loss
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })

        # Calculate epoch metrics
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model on the validation set.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for batch in pbar:
                    # Move batch to device
                    frames = batch['frames'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass
                    if 'landmarks' in batch and hasattr(self.model, 'forward') and 'landmarks' in self.model.forward.__code__.co_varnames:
                        landmarks = batch['landmarks'].to(self.device)
                        outputs = self.model(frames, landmarks)
                    else:
                        outputs = self.model(frames)

                    # Get logits
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    # Calculate loss
                    loss = self.criterion(logits, labels)

                    # Update metrics
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{100 * correct / total:.2f}%"
                    })

        # Calculate epoch metrics
        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Move model to device
        self.model.to(self.device)

        # Initialize history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")

            # Train for one epoch
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate if scheduler is available
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed in {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Check if this is the best model
            is_best = False
            if val_acc > self.best_val_acc:
                logger.info(f"Validation accuracy improved from {self.best_val_acc:.4f} to {val_acc:.4f}")
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epochs")

            # Save checkpoint
            self._save_checkpoint(epoch, is_best)

            # Early stopping
            if self.early_stopping > 0 and self.epochs_without_improvement >= self.early_stopping:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

        # Close tensorboard writer
        self.writer.close()

        # Log training summary
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}")

        return history

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'val_acc': self.best_val_acc
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'model_best.pth')
            torch.save(checkpoint, best_path)

            # Save model architecture as text
            with open(os.path.join(self.output_dir, 'model_architecture.txt'), 'w') as f:
                f.write(str(self.model))


def train_model(
    model_name: str,
    train_csv: str,
    val_csv: str,
    data_dir: str,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """
    Train a sign language recognition model.

    Args:
        model_name: Name of the model to train
        train_csv: Path to training data CSV
        val_csv: Path to validation data CSV
        data_dir: Directory containing the data
        output_dir: Directory to save outputs
        config: Configuration dictionary
    """
    # Extract configuration
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    num_classes = config.get('num_classes', 100)
    use_landmarks = config.get('use_landmarks', True)
    early_stopping = config.get('early_stopping', 10)
    max_seq_len = config.get('max_seq_len', 60)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets and dataloaders
    train_dataset = SignVideoDataset(
        train_csv,
        data_dir,
        use_landmarks=use_landmarks,
        max_seq_len=max_seq_len
    )
    val_dataset = SignVideoDataset(
        val_csv,
        data_dir,
        use_landmarks=use_landmarks,
        max_seq_len=max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Create model
    model = create_model(
        model_name,
        num_classes=num_classes,
        **config.get('model_params', {})
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        num_epochs=num_epochs,
        early_stopping=early_stopping
    )

    # Train model
    history = trainer.train()

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train sign language recognition model')
    parser.add_argument('--model', type=str, default='cnn_lstm', help='Model name')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation data CSV')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config JSON file')

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_classes': 100,
            'use_landmarks': True,
            'early_stopping': 10,
            'max_seq_len': 60,
            'model_params': {
                'frame_dim': (3, 224, 224),
                'lstm_hidden_dim': 512,
                'lstm_layers': 2,
                'dropout': 0.5,
                'bidirectional': True,
                'attention': True
            }
        }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Train model
    train_model(
        args.model,
        args.train_csv,
        args.val_csv,
        args.data_dir,
        args.output_dir,
        config
    )
