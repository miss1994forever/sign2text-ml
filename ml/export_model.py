#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export a trained PyTorch or TensorFlow model to Core ML format.
This script handles the conversion of deep learning models trained for sign language
translation to Core ML format for use in iOS applications.
"""

import os
import argparse
import logging
import json
from pathlib import Path

import torch
import numpy as np
import coremltools as ct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pytorch_model(model_path, model_class=None):
    """
    Load a PyTorch model from the specified path.

    Args:
        model_path: Path to the saved PyTorch model (.pt or .pth file)
        model_class: Optional class reference to initialize the model

    Returns:
        Loaded PyTorch model
    """
    logger.info(f"Loading PyTorch model from {model_path}")

    if model_class is not None:
        # If model class is provided, initialize the model first
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        # Direct load for models saved with torch.save(model)
        model = torch.load(model_path, map_location="cpu")

    model.eval()
    return model


def convert_pytorch_to_coreml(model, input_shape, output_names, model_name="SignLanguageTranslator"):
    """
    Convert a PyTorch model to Core ML format.

    Args:
        model: PyTorch model
        input_shape: Shape of the input tensor (e.g., [1, 3, 224, 224])
        output_names: List of names for the output tensors
        model_name: Name for the Core ML model

    Returns:
        Core ML model
    """
    logger.info(f"Converting PyTorch model to Core ML format with input shape {input_shape}")

    # Create a dummy input
    dummy_input = torch.rand(*input_shape)

    # Convert to Core ML
    model_coreml = ct.convert(
        model,
        inputs=[ct.TensorType(shape=input_shape, name="input")],
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # Set model metadata
    model_coreml.short_description = f"Sign Language Translation Model ({model_name})"
    model_coreml.author = "Sign2Text Team"
    model_coreml.license = "All rights reserved"
    model_coreml.version = "1.0"

    return model_coreml


def load_tensorflow_model(model_path):
    """
    Load a TensorFlow model from the specified path.

    Args:
        model_path: Path to the saved TensorFlow model directory

    Returns:
        Loaded TensorFlow model
    """
    import tensorflow as tf

    logger.info(f"Loading TensorFlow model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def convert_tensorflow_to_coreml(model, input_shape, output_names, model_name="SignLanguageTranslator"):
    """
    Convert a TensorFlow model to Core ML format.

    Args:
        model: TensorFlow model
        input_shape: Shape of the input tensor (e.g., [1, 224, 224, 3])
        output_names: List of names for the output tensors
        model_name: Name for the Core ML model

    Returns:
        Core ML model
    """
    logger.info(f"Converting TensorFlow model to Core ML format with input shape {input_shape}")

    # Convert to Core ML
    model_coreml = ct.convert(
        model,
        inputs=[ct.TensorType(shape=input_shape, name="input")],
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )

    # Set model metadata
    model_coreml.short_description = f"Sign Language Translation Model ({model_name})"
    model_coreml.author = "Sign2Text Team"
    model_coreml.license = "All rights reserved"
    model_coreml.version = "1.0"

    return model_coreml


def add_model_metadata(model, config_path=None):
    """
    Add metadata to the model from a config file.

    Args:
        model: Core ML model
        config_path: Path to a JSON config file with metadata

    Returns:
        Core ML model with added metadata
    """
    if config_path and os.path.exists(config_path):
        logger.info(f"Adding metadata from {config_path}")
        with open(config_path, 'r') as f:
            metadata = json.load(f)

        # Add model metadata
        for key, value in metadata.get('model_info', {}).items():
            setattr(model, key, value)

        # Add user-defined metadata
        for key, value in metadata.get('user_defined', {}).items():
            model.user_defined[key] = str(value)

    return model


def main():
    parser = argparse.ArgumentParser(description="Convert trained models to Core ML format")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], default="pytorch",
                        help="Deep learning framework used for the model")
    parser.add_argument("--output_path", help="Path to save the Core ML model")
    parser.add_argument("--input_shape", type=str, default="1,3,224,224",
                        help="Shape of the input tensor (comma separated)")
    parser.add_argument("--output_names", type=str, default="output",
                        help="Comma separated names for output tensors")
    parser.add_argument("--model_name", type=str, default="SignLanguageTranslator",
                        help="Name for the Core ML model")
    parser.add_argument("--config", help="Path to a JSON config file with model metadata")

    args = parser.parse_args()

    # Parse input shape and output names
    input_shape = [int(dim) for dim in args.input_shape.split(",")]
    output_names = args.output_names.split(",")

    # Set default output path if not provided
    if not args.output_path:
        output_dir = os.path.dirname(args.model_path)
        args.output_path = os.path.join(output_dir, f"{args.model_name}.mlpackage")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        # Load and convert the model based on the framework
        if args.framework == "pytorch":
            model = load_pytorch_model(args.model_path)
            model_coreml = convert_pytorch_to_coreml(model, input_shape, output_names, args.model_name)
        else:  # tensorflow
            model = load_tensorflow_model(args.model_path)
            model_coreml = convert_tensorflow_to_coreml(model, input_shape, output_names, args.model_name)

        # Add metadata
        model_coreml = add_model_metadata(model_coreml, args.config)

        # Save the Core ML model
        logger.info(f"Saving Core ML model to {args.output_path}")
        model_coreml.save(args.output_path)
        logger.info("Model conversion completed successfully")

    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        raise


if __name__ == "__main__":
    main()
