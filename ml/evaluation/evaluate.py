#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation module for the Sign2Text ML pipeline.

This module provides functions to evaluate trained sign language recognition models,
including accuracy metrics, confusion matrix analysis, and Word Error Rate (WER) calculation.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from jiwer import wer

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.cnn_lstm import create_model
from training.trainer import SignVideoDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator class for sign language recognition models."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: str = 'cuda',
        output_dir: str = './evaluation_results',
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            test_loader: DataLoader for test data
            criterion: Loss function
            device: Device to use for evaluation
            output_dir: Directory to save evaluation results
            class_names: List of class names for better reporting
        """
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.class_names = class_names

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Evaluator initialized with {len(test_loader.dataset)} test samples")
        logger.info(f"Using device: {device}")

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_confidences = []
        inference_times = []

        logger.info("Starting evaluation...")

        with torch.no_grad():
            with tqdm(self.test_loader, desc="Evaluating") as pbar:
                for batch in pbar:
                    # Move batch to device
                    frames = batch['frames'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Measure inference time
                    start_time = time.time()

                    # Forward pass
                    if 'landmarks' in batch and hasattr(self.model, 'forward') and 'landmarks' in self.model.forward.__code__.co_varnames:
                        landmarks = batch['landmarks'].to(self.device)
                        outputs = self.model(frames, landmarks)
                    else:
                        outputs = self.model(frames)

                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)

                    # Get logits
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    # Calculate loss
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()

                    # Get predictions and confidences
                    probabilities = torch.softmax(logits, dim=1)
                    confidences, predictions = torch.max(probabilities, dim=1)

                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_confidences.extend(confidences.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_loss = total_loss / len(self.test_loader)
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds

        # Create classification report
        if self.class_names:
            target_names = self.class_names
        else:
            target_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]

        class_report = classification_report(
            all_labels,
            all_predictions,
            target_names=target_names,
            output_dict=True
        )

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Log summary
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        logger.info(f"Average inference time: {avg_inference_time:.2f} ms per sample")

        # Store results
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'inference_time_ms': avg_inference_time,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'confidences': all_confidences
        }

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results.

        Args:
            results: Dictionary with evaluation metrics
        """
        # Save metrics to JSON
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                'accuracy': float(results['accuracy']),
                'loss': float(results['loss']),
                'inference_time_ms': float(results['inference_time_ms']),
                'classification_report': results['classification_report']
            }
            json.dump(serializable_results, f, indent=4)

        # Create and save confusion matrix plot
        self._plot_confusion_matrix(
            results['confusion_matrix'],
            self.class_names if self.class_names else None
        )

        # Create and save confidence distribution
        self._plot_confidence_distribution(
            results['confidences'],
            results['predictions'],
            results['labels']
        )

        logger.info(f"Evaluation results saved to {self.output_dir}")

    def _plot_confusion_matrix(
        self,
        confusion_mat: List[List[int]],
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Plot and save the confusion matrix.

        Args:
            confusion_mat: Confusion matrix data
            class_names: Class names for axis labels
        """
        plt.figure(figsize=(12, 10))

        cm = np.array(confusion_mat)

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create labels if not provided
        if not class_names:
            class_names = [f"Class {i}" for i in range(len(cm))]

        # Create heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

    def _plot_confidence_distribution(
        self,
        confidences: List[float],
        predictions: List[int],
        labels: List[int]
    ) -> None:
        """
        Plot and save the confidence distribution.

        Args:
            confidences: Model confidence scores
            predictions: Model predictions
            labels: True labels
        """
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Separate correct and incorrect predictions
        correct_mask = predictions == labels
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]

        plt.figure(figsize=(10, 6))

        # Plot histograms
        plt.hist(
            correct_conf,
            bins=20,
            alpha=0.5,
            label=f'Correct Predictions ({len(correct_conf)})',
            color='green'
        )
        plt.hist(
            incorrect_conf,
            bins=20,
            alpha=0.5,
            label=f'Incorrect Predictions ({len(incorrect_conf)})',
            color='red'
        )

        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Distribution of Model Confidence')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'confidence_distribution.png'), dpi=300)
        plt.close()


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate between predicted and reference texts.

    Args:
        predictions: List of predicted text strings
        references: List of reference text strings

    Returns:
        Word Error Rate score
    """
    return wer(references, predictions)


def evaluate_sign_to_text_quality(
    model_outputs: List[str],
    ground_truth: List[str],
    output_dir: str = './evaluation_results'
) -> Dict[str, float]:
    """
    Evaluate the quality of sign language to text translation.

    Args:
        model_outputs: List of model's text outputs
        ground_truth: List of ground truth text translations
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate Word Error Rate
    wer_score = calculate_wer(model_outputs, ground_truth)

    # Calculate character-level metrics
    char_errors = 0
    char_total = 0

    for pred, ref in zip(model_outputs, ground_truth):
        # Character Error Rate calculation
        import editdistance
        distance = editdistance.eval(pred, ref)
        char_errors += distance
        char_total += len(ref)

    cer = char_errors / max(1, char_total)

    # Calculate BLEU score if nltk is available
    bleu_score = None
    try:
        from nltk.translate.bleu_score import corpus_bleu

        # Tokenize predictions and references
        tokenized_preds = [pred.split() for pred in model_outputs]
        tokenized_refs = [[ref.split()] for ref in ground_truth]  # corpus_bleu expects list of list of references

        # Calculate BLEU
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds)
    except ImportError:
        logger.warning("NLTK not available. BLEU score not calculated.")

    # Store results
    results = {
        'wer': wer_score,
        'cer': cer
    }

    if bleu_score is not None:
        results['bleu'] = bleu_score

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'text_quality_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Log summary
    logger.info(f"Text quality evaluation completed. WER: {wer_score:.4f}, CER: {cer:.4f}")
    if bleu_score is not None:
        logger.info(f"BLEU score: {bleu_score:.4f}")

    return results


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    model_config: Dict[str, Any],
    device: str = 'cuda'
) -> nn.Module:
    """
    Load a model from checkpoint.

    Args:
        model_name: Name of the model architecture
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration dictionary
        device: Device to load the model on

    Returns:
        Loaded model
    """
    # Create model
    model = create_model(model_name, **model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    # Load state dictionary
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}")

    return model


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    test_csv: str,
    data_dir: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a trained sign language recognition model.

    Args:
        model_name: Name of the model to evaluate
        checkpoint_path: Path to the model checkpoint
        test_csv: Path to test data CSV
        data_dir: Directory containing the data
        output_dir: Directory to save outputs
        config: Configuration dictionary

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract configuration
    batch_size = config.get('batch_size', 32)
    num_classes = config.get('num_classes', 100)
    use_landmarks = config.get('use_landmarks', True)
    max_seq_len = config.get('max_seq_len', 60)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test dataset and dataloader
    test_dataset = SignVideoDataset(
        test_csv,
        data_dir,
        use_landmarks=use_landmarks,
        max_seq_len=max_seq_len
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Get class names
    class_names = test_dataset.classes

    # Create model and load from checkpoint
    model_config = {
        'num_classes': num_classes,
        **config.get('model_params', {})
    }
    model = load_model_from_checkpoint(model_name, checkpoint_path, model_config, device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        class_names=class_names
    )

    results = evaluator.evaluate()

    # Convert numpy arrays to Python native types for returning
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate sign language recognition model')
    parser.add_argument('--model', type=str, default='cnn_lstm', help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config JSON file')

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'batch_size': 32,
            'num_classes': 100,
            'use_landmarks': True,
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

    # Evaluate model
    results = evaluate_model(
        args.model,
        args.checkpoint,
        args.test_csv,
        args.data_dir,
        args.output_dir,
        config
    )

    logger.info("Evaluation completed.")
