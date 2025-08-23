#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examine .mat files in the data directory to understand their structure.
This script helps analyze the content and organization of MATLAB data files.
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
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def examine_mat_file(file_path):
    """
    Examine the contents of a single .mat file.

    Args:
        file_path: Path to the .mat file

    Returns:
        Dictionary with information about the file
    """
    try:
        # Load the .mat file
        mat_data = loadmat(file_path)

        # Extract directory name (potential class label)
        class_name = os.path.basename(os.path.dirname(file_path))

        # Filter out internal MATLAB keys
        relevant_keys = [k for k in mat_data.keys() if not k.startswith('__')]

        # Extract information about each key
        key_info = {}
        for key in relevant_keys:
            if isinstance(mat_data[key], np.ndarray):
                key_info[key] = {
                    'type': str(mat_data[key].dtype),
                    'shape': mat_data[key].shape,
                    'min': float(np.min(mat_data[key])) if mat_data[key].size > 0 else None,
                    'max': float(np.max(mat_data[key])) if mat_data[key].size > 0 else None,
                    'mean': float(np.mean(mat_data[key])) if mat_data[key].size > 0 else None,
                    'has_nan': bool(np.isnan(mat_data[key]).any()) if mat_data[key].size > 0 else None
                }
            else:
                key_info[key] = {
                    'type': type(mat_data[key]).__name__
                }

        return {
            'file_path': file_path,
            'class_name': class_name,
            'keys': relevant_keys,
            'key_info': key_info
        }

    except Exception as e:
        logger.error(f"Error examining {file_path}: {e}")
        return {
            'file_path': file_path,
            'error': str(e)
        }

def check_mat_files(data_dir, output_dir=None, num_files=10, random_sample=False,
                    save_summary=True, plot_data=True):
    """
    Check .mat files in a directory and provide a summary of their structure.

    Args:
        data_dir: Directory containing .mat files
        output_dir: Directory to save results (if None, don't save)
        num_files: Number of files to check
        random_sample: Whether to select files randomly
        save_summary: Whether to save the summary information
        plot_data: Whether to create and save plots

    Returns:
        Dictionary with summary information
    """
    data_path = Path(data_dir)

    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

    # Find all .mat files
    mat_files = list(data_path.glob('**/*.mat'))
    logger.info(f'Found {len(mat_files)} .mat files in {data_dir}')

    if len(mat_files) == 0:
        logger.error("No .mat files found!")
        return None

    # Select files to examine
    if num_files >= len(mat_files):
        selected_files = mat_files
        logger.info(f"Examining all {len(selected_files)} files")
    elif random_sample:
        selected_files = np.random.choice(mat_files, num_files, replace=False)
        logger.info(f"Examining {num_files} randomly selected files")
    else:
        selected_files = mat_files[:num_files]
        logger.info(f"Examining first {num_files} files")

    # Examine each file
    results = []
    for file_path in tqdm(selected_files, desc="Examining files"):
        file_info = examine_mat_file(file_path)
        results.append(file_info)

    # Extract class information
    classes = {}
    for result in results:
        class_name = result.get('class_name')
        if class_name:
            classes[class_name] = classes.get(class_name, 0) + 1

    # Extract key information
    key_presence = {}
    for result in results:
        if 'keys' in result:
            for key in result['keys']:
                key_presence[key] = key_presence.get(key, 0) + 1

    # Summarize results
    summary = {
        'total_files': len(mat_files),
        'files_examined': len(results),
        'num_classes': len(classes),
        'classes': classes,
        'keys_found': key_presence,
        'results': results
    }

    # Print summary
    logger.info("\n--- SUMMARY ---")
    logger.info(f"Total files found: {summary['total_files']}")
    logger.info(f"Files examined: {summary['files_examined']}")
    logger.info(f"Number of classes (directories): {summary['num_classes']}")
    logger.info("Top 10 classes by frequency:")
    for class_name, count in sorted(classes.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {class_name}: {count} files")

    logger.info("\nKeys found in .mat files:")
    for key, count in sorted(key_presence.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {key}: present in {count}/{summary['files_examined']} files ({count/summary['files_examined']*100:.1f}%)")

    # Look at the first file with the most common key to understand its structure
    most_common_key = max(key_presence.items(), key=lambda x: x[1])[0] if key_presence else None

    if most_common_key:
        logger.info(f"\nExamining structure of most common key: '{most_common_key}'")
        for result in results:
            if 'key_info' in result and most_common_key in result.get('key_info', {}):
                info = result['key_info'][most_common_key]
                logger.info(f"Example from file: {os.path.basename(result['file_path'])}")
                for k, v in info.items():
                    logger.info(f"  {k}: {v}")
                break

    # Save summary if requested
    if save_summary and output_dir:
        # Create a DataFrame for the basic file information
        file_info_list = []
        for result in results:
            if 'error' not in result:
                file_info = {
                    'filename': os.path.basename(result['file_path']),
                    'class_name': result['class_name'],
                    'num_keys': len(result['keys'])
                }

                # Add information about presence of each common key
                for key in key_presence:
                    file_info[f'has_{key}'] = key in result['keys']

                # Add shape information for the most common key
                if most_common_key in result.get('key_info', {}):
                    file_info[f'{most_common_key}_shape'] = str(result['key_info'][most_common_key]['shape'])

                file_info_list.append(file_info)

        if file_info_list:
            df = pd.DataFrame(file_info_list)
            csv_path = os.path.join(output_dir, 'mat_files_summary.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved summary to {csv_path}")

    # Create plots if requested
    if plot_data and output_dir:
        # Plot class distribution
        plt.figure(figsize=(12, 6))
        classes_df = pd.Series(classes).reset_index()
        classes_df.columns = ['class', 'count']
        classes_df = classes_df.sort_values('count', ascending=False).head(20)  # Top 20 classes

        sns.barplot(x='class', y='count', data=classes_df)
        plt.title('Class Distribution (Top 20)')
        plt.xlabel('Class')
        plt.ylabel('Number of Files')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
        plt.close()

        # Plot key presence
        plt.figure(figsize=(10, 6))
        keys_df = pd.DataFrame(list(key_presence.items()), columns=['key', 'count'])
        keys_df = keys_df.sort_values('count', ascending=False)

        sns.barplot(x='key', y='count', data=keys_df)
        plt.title('Keys Present in .mat Files')
        plt.xlabel('Key')
        plt.ylabel('Number of Files')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'key_presence.png'), dpi=300)
        plt.close()

        # If we have data arrays, plot some statistics
        if most_common_key:
            # Collect shapes and values
            shapes = []
            values = []

            for result in results:
                if 'key_info' in result and most_common_key in result.get('key_info', {}):
                    shape_str = str(result['key_info'][most_common_key]['shape'])
                    shapes.append(shape_str)

                    # For a sample of files, load the actual data for histogram
                    if len(values) < 10:  # Limit to 10 files for memory reasons
                        try:
                            mat_data = loadmat(result['file_path'])
                            data_values = mat_data[most_common_key].flatten()
                            values.extend(data_values[:1000])  # Take a subset of values
                        except:
                            pass

            # Plot shape distribution
            plt.figure(figsize=(10, 6))
            shape_counts = pd.Series(shapes).value_counts()
            sns.barplot(x=shape_counts.index, y=shape_counts.values)
            plt.title(f'Array Shapes for Key: {most_common_key}')
            plt.xlabel('Shape')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'array_shapes.png'), dpi=300)
            plt.close()

            # Plot value histogram
            if values:
                plt.figure(figsize=(10, 6))
                sns.histplot(values, kde=True)
                plt.title(f'Value Distribution for Key: {most_common_key}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'value_histogram.png'), dpi=300)
                plt.close()

    return summary

def main():
    parser = argparse.ArgumentParser(description='Check .mat files in a directory')
    parser.add_argument('--data_dir', required=True, help='Directory containing .mat files')
    parser.add_argument('--output_dir', help='Directory to save results')
    parser.add_argument('--num_files', type=int, default=10, help='Number of files to check')
    parser.add_argument('--random_sample', action='store_true', help='Select files randomly')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')

    args = parser.parse_args()

    check_mat_files(
        args.data_dir,
        args.output_dir,
        args.num_files,
        args.random_sample,
        plot_data=not args.no_plot
    )

if __name__ == "__main__":
    main()
