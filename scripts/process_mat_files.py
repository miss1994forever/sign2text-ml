#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process .mat files from the data directory and prepare them for ML training.
This script analyzes .mat files, extracts features, and organizes them for the Sign2Text pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_mat_files(data_dir: str, output_dir: str, sample_limit: int = None) -> pd.DataFrame:
    """
    Analyze .mat files to understand their structure.

    Args:
        data_dir: Directory containing .mat files
        output_dir: Directory to save analysis results
        sample_limit: Maximum number of files to analyze (None for all)

    Returns:
        DataFrame with analysis results
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Find all .mat files
    mat_files = list(data_path.glob('**/*.mat'))
    logger.info(f'Found {len(mat_files)} .mat files')

    if sample_limit and sample_limit < len(mat_files):
        mat_files = mat_files[:sample_limit]
        logger.info(f'Analyzing {sample_limit} files as a sample')

    # Process files
    sample_data = []
    for file_path in tqdm(mat_files, desc="Analyzing .mat files"):
        try:
            # Extract class from directory name
            class_name = file_path.parent.name

            # Load .mat file
            mat_data = loadmat(file_path)

            # Extract keys and data shapes
            info = {
                'file_path': str(file_path),
                'filename': file_path.name,
                'class_name': class_name,
                'keys': list(k for k in mat_data.keys() if not k.startswith('_'))
            }

            # Add shape information for each key
            for key in info['keys']:
                if key in mat_data and hasattr(mat_data[key], 'shape'):
                    info[f'{key}_shape'] = str(mat_data[key].shape)

            sample_data.append(info)

        except Exception as e:
            logger.error(f'Error processing {file_path}: {e}')

    # Create DataFrame with the sample data
    if not sample_data:
        logger.error("No files were successfully processed!")
        return None

    df = pd.DataFrame(sample_data)

    # Save the analysis results
    analysis_path = output_path / 'mat_files_analysis.csv'
    df.to_csv(analysis_path, index=False)
    logger.info(f'Saved analysis to {analysis_path}')

    # Print summary
    logger.info('\nData summary:')
    logger.info(f'Classes found: {df.class_name.nunique()}')

    # Fix: Use df['keys'] instead of df.keys (which is a method)
    if 'keys' in df.columns:
        all_keys = set().union(*[set(k) for k in df['keys']])
        logger.info(f'Unique keys in .mat files: {all_keys}')

        # Identify common data structure
        common_keys = set.intersection(*[set(k) for k in df['keys']])
        logger.info(f'Common keys across all files: {common_keys}')

    return df

def extract_features(data_dir: str, output_dir: str, key_mapping: dict, class_mapping_file: str = None) -> pd.DataFrame:
    """
    Extract features from .mat files and save them in a format ready for ML pipeline.

    Args:
        data_dir: Directory containing .mat files
        output_dir: Directory to save processed data
        key_mapping: Dictionary mapping mat file keys to feature names
        class_mapping_file: Optional file containing class name mapping

    Returns:
        DataFrame with feature file paths and labels
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load class mapping if provided
    class_mapping = {}
    if class_mapping_file and os.path.exists(class_mapping_file):
        try:
            df_mapping = pd.read_csv(class_mapping_file)
            class_mapping = dict(zip(df_mapping['original'], df_mapping['mapped']))
            logger.info(f"Loaded class mapping with {len(class_mapping)} entries")
        except Exception as e:
            logger.error(f"Error loading class mapping file: {e}")

    # Find all .mat files
    mat_files = list(data_path.glob('**/*.mat'))
    logger.info(f'Found {len(mat_files)} .mat files to process')

    # Process files
    processed_data = []

    for file_path in tqdm(mat_files, desc="Processing .mat files"):
        try:
            # Extract class from directory name
            original_class = file_path.parent.name

            # Map class if mapping exists
            if original_class in class_mapping:
                class_name = class_mapping[original_class]
            else:
                class_name = original_class

            # Load .mat file
            mat_data = loadmat(file_path)

            # Extract features based on key mapping
            features = {}
            for mat_key, feature_name in key_mapping.items():
                if mat_key in mat_data:
                    features[feature_name] = mat_data[mat_key]

            if not features:
                logger.warning(f"No matching keys found in {file_path}")
                continue

            # Create a unique output filename
            output_filename = f"{original_class}_{file_path.stem}.npz"
            output_file_path = output_path / output_filename

            # Save extracted features
            np.savez_compressed(
                output_file_path,
                **features,
                original_class=original_class,
                class_name=class_name
            )

            # Add to processed data
            processed_data.append({
                'original_file': str(file_path),
                'feature_path': str(output_file_path.relative_to(output_path.parent)),
                'class_name': class_name,
                'original_class': original_class
            })

        except Exception as e:
            logger.error(f'Error processing {file_path}: {e}')

    # Create DataFrame with the processed data
    if not processed_data:
        logger.error("No files were successfully processed!")
        return None

    df = pd.DataFrame(processed_data)

    # Save the dataset info
    dataset_info_path = output_path / 'dataset_info.csv'
    df.to_csv(dataset_info_path, index=False)
    logger.info(f'Saved dataset info to {dataset_info_path}')

    # Print summary
    logger.info('\nProcessing summary:')
    logger.info(f'Total processed files: {len(df)}')
    logger.info(f'Classes: {df.class_name.nunique()}')

    return df

def create_class_mapping(analysis_df: pd.DataFrame, output_file: str) -> None:
    """
    Create a template for class mapping based on analysis results.

    Args:
        analysis_df: DataFrame with analysis results
        output_file: Path to save the mapping template
    """
    unique_classes = sorted(analysis_df['class_name'].unique())

    # Create mapping DataFrame
    mapping_df = pd.DataFrame({
        'original': unique_classes,
        'mapped': unique_classes,  # Default to same as original
        'description': [''] * len(unique_classes)
    })

    # Save mapping template
    mapping_df.to_csv(output_file, index=False)
    logger.info(f'Created class mapping template at {output_file}')
    logger.info('Edit this file to map class names to your preferred labels')

def main():
    parser = argparse.ArgumentParser(description='Process .mat files for sign language recognition')
    parser.add_argument('--data_dir', required=True, help='Directory containing .mat files')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed data')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze file structure without processing')
    parser.add_argument('--sample_limit', type=int, default=100, help='Maximum number of files to analyze (for --analyze_only)')
    parser.add_argument('--class_mapping', help='Path to class mapping CSV file')
    parser.add_argument('--create_class_mapping', action='store_true', help='Create a class mapping template')
    parser.add_argument('--key_mapping', help='Key mapping JSON string, e.g. \'{"data":"features", "label":"target"}\'')

    args = parser.parse_args()

    # Analyze files
    if args.analyze_only or args.create_class_mapping:
        analysis_df = analyze_mat_files(args.data_dir, args.output_dir, args.sample_limit)

        if args.create_class_mapping and analysis_df is not None:
            mapping_file = os.path.join(args.output_dir, 'class_mapping_template.csv')
            create_class_mapping(analysis_df, mapping_file)

        return

    # Process files
    if not args.key_mapping:
        logger.error("--key_mapping is required for processing. Use --analyze_only first to understand file structure.")
        return

    try:
        import json
        key_mapping = json.loads(args.key_mapping)
    except json.JSONDecodeError:
        logger.error("Invalid key_mapping JSON format")
        return

    extract_features(args.data_dir, args.output_dir, key_mapping, args.class_mapping)

if __name__ == "__main__":
    main()
