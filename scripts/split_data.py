#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split processed sign language data into training, validation, and test sets.
This script ensures proper stratification and data distribution for ML training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_dataset(
    dataset_file: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42
) -> dict:
    """
    Split dataset into training, validation, and test sets.

    Args:
        dataset_file: Path to the CSV file with dataset information
        output_dir: Directory to save the split datasets
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        stratify: Whether to perform stratified split based on class labels
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with DataFrames for each split
    """
    # Verify ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        logger.warning(f"Split ratios sum to {total_ratio}, not 1.0. Normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    # Load dataset
    try:
        df = pd.read_csv(dataset_file)
    except Exception as e:
        logger.error(f"Error loading dataset file: {e}")
        return None

    logger.info(f"Loaded {len(df)} samples from {dataset_file}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Check if stratification is possible
    use_stratify = False
    stratify_col = None
    if stratify and 'class_name' in df.columns:
        unique_classes = df['class_name'].unique()
        num_classes = len(unique_classes)

        # Calculate minimum split size
        min_split_size = int(len(df) * min(val_ratio, test_ratio))

        logger.info(f"Using stratified split based on {num_classes} classes")

        # Check if there are more classes than samples in the smallest split
        if num_classes > min_split_size:
            logger.warning(f"Cannot perform stratified split: number of classes ({num_classes}) "
                         f"exceeds the minimum split size ({min_split_size})")
            logger.warning("Falling back to random (non-stratified) splitting")
        else:
            use_stratify = True
            stratify_col = df['class_name']
    elif stratify:
        logger.warning("Stratification requested but 'class_name' column not found. Using random split.")

    # First split: train vs. (val + test)
    val_test_ratio = val_ratio + test_ratio
    relative_val_ratio = val_ratio / val_test_ratio

    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=stratify_col if use_stratify else None
    )

    # Second split: val vs. test
    stratify_for_val_test = None
    if use_stratify and 'class_name' in temp_df.columns:
        stratify_for_val_test = temp_df['class_name']

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / val_test_ratio),
        random_state=random_seed,
        stratify=stratify_for_val_test
    )

    # Save the splits
    train_path = output_path / 'train_set.csv'
    val_path = output_path / 'val_set.csv'
    test_path = output_path / 'test_set.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Data split complete:")
    logger.info(f"  Training set: {len(train_df)} samples ({len(train_df)/len(df):.1%}) - saved to {train_path}")
    logger.info(f"  Validation set: {len(val_df)} samples ({len(val_df)/len(df):.1%}) - saved to {val_path}")
    logger.info(f"  Test set: {len(test_df)} samples ({len(test_df)/len(df):.1%}) - saved to {test_path}")

    # Check class distribution in splits
    if 'class_name' in df.columns:
        logger.info("Class distribution in splits:")

        total_class_dist = df['class_name'].value_counts(normalize=True)
        train_class_dist = train_df['class_name'].value_counts(normalize=True)
        val_class_dist = val_df['class_name'].value_counts(normalize=True)
        test_class_dist = test_df['class_name'].value_counts(normalize=True)

        # Log distribution for the top 5 classes
        top_classes = total_class_dist.head(5).index
        for cls in top_classes:
            logger.info(f"  Class '{cls}':")
            logger.info(f"    Total: {total_class_dist.get(cls, 0):.1%}")
            logger.info(f"    Train: {train_class_dist.get(cls, 0):.1%}")
            logger.info(f"    Val: {val_class_dist.get(cls, 0):.1%}")
            logger.info(f"    Test: {test_class_dist.get(cls, 0):.1%}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def create_subset(
    dataset_file: str,
    output_file: str,
    subset_size: int = 1000,
    stratify: bool = True,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a smaller subset of the dataset for quick testing/development.

    Args:
        dataset_file: Path to the CSV file with dataset information
        output_file: Path to save the subset dataset
        subset_size: Number of samples in the subset
        stratify: Whether to preserve class distribution
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing the subset
    """
    try:
        df = pd.read_csv(dataset_file)
    except Exception as e:
        logger.error(f"Error loading dataset file: {e}")
        return None

    if len(df) <= subset_size:
        logger.warning(f"Dataset only has {len(df)} samples, less than requested {subset_size}. Using entire dataset.")
        subset_df = df
    else:
        # Perform stratified sampling if requested
        if stratify and 'class_name' in df.columns:
            # Calculate number of samples per class to maintain distribution
            class_counts = df['class_name'].value_counts(normalize=True)

            subset_df = pd.DataFrame()
            for class_name, proportion in class_counts.items():
                class_df = df[df['class_name'] == class_name]
                class_samples = min(
                    max(1, int(subset_size * proportion)),  # At least 1 sample
                    len(class_df)  # But not more than available
                )

                sampled = class_df.sample(n=class_samples, random_state=random_seed)
                subset_df = pd.concat([subset_df, sampled])

            # If we didn't get enough samples due to rounding, add more from largest classes
            if len(subset_df) < subset_size:
                remaining = subset_size - len(subset_df)
                largest_classes = class_counts.index[:5]  # Top 5 largest classes
                extra_samples = df[
                    (df['class_name'].isin(largest_classes)) &
                    (~df.index.isin(subset_df.index))
                ].sample(n=min(remaining, len(df) - len(subset_df)), random_state=random_seed)

                subset_df = pd.concat([subset_df, extra_samples])
        else:
            # Simple random sampling
            subset_df = df.sample(n=subset_size, random_state=random_seed)

    # Save the subset
    subset_df.to_csv(output_file, index=False)
    logger.info(f"Created subset with {len(subset_df)} samples, saved to {output_file}")

    return subset_df

def main():
    parser = argparse.ArgumentParser(description='Split sign language dataset into train/val/test sets')
    parser.add_argument('--dataset_file', required=True, help='Path to the CSV file with dataset information')
    parser.add_argument('--output_dir', required=True, help='Directory to save the split datasets')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Proportion of data for testing')
    parser.add_argument('--no_stratify', action='store_true', help='Disable stratified splitting')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--create_subset', action='store_true', help='Create a smaller subset for development')
    parser.add_argument('--subset_size', type=int, default=1000, help='Number of samples in the subset')

    args = parser.parse_args()

    # Create subset if requested
    if args.create_subset:
        subset_file = os.path.join(args.output_dir, 'development_subset.csv')
        create_subset(
            args.dataset_file,
            subset_file,
            args.subset_size,
            not args.no_stratify,
            args.random_seed
        )

        # Use the subset for splitting if it was created
        dataset_file = subset_file
        logger.info(f"Using subset file for train/val/test splitting: {subset_file}")
    else:
        dataset_file = args.dataset_file

    # Split the dataset
    split_dataset(
        dataset_file,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        not args.no_stratify,
        args.random_seed
    )

if __name__ == "__main__":
    main()
