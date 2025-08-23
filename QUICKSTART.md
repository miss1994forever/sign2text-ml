# Sign2Text ML Pipeline: Quick Start Guide

This guide will help you quickly get started with the Sign2Text ML pipeline for sign language recognition.

## 1. Requirements

Before you begin, ensure you have:

- Python 3.8+ installed
- [Task](https://taskfile.dev/) installed (optional but recommended)
- CUDA-compatible GPU (recommended for training)

## 2. Installation

First, install the required dependencies:

```bash
# Using Task
task install

# Or with pip directly
pip install -r ml/requirements.txt
```

## 3. Data Analysis & Preparation

Start by analyzing your .mat files to understand their structure:

```bash
task analyze-data
```

This will create a summary CSV file in `processed_data/mat_files_analysis.csv` that shows the structure of your data files.

Next, create a class mapping template:

```bash
task create-class-mapping
```

Edit the generated `processed_data/class_mapping_template.csv` file to map your numeric class IDs to meaningful sign names.

## 4. Preprocess Data

Preprocess the .mat files into a format suitable for training:

```bash
task preprocess
```

If your .mat files have different key names, customize the command:

```bash
task preprocess KEY_MAPPING='{"your_data_key":"features", "your_label_key":"target"}'
```

Split the processed data into train/validation/test sets:

```bash
task split-data
```

## 5. Training

### Quick Development Iteration

For quick testing of your pipeline, create a small development subset and train on it:

```bash
task create-dev-subset
task train-dev
```

This will create a smaller dataset and train a simplified model for rapid experimentation.

### Full Model Training

Train the full CNN-LSTM model:

```bash
task train
```

Or train the model with hand pose features:

```bash
task train-handpose
```

## 6. Evaluation

Evaluate your trained model:

```bash
task evaluate
```

To evaluate a specific model:

```bash
task evaluate MODEL_PATH=./models/my_model/model_best.pth MODEL_NAME=my_model
```

## 7. Export for Deployment

Export your trained model to Core ML format for iOS deployment:

```bash
task export MODEL_PATH=./models/my_model/model_best.pth
```

## 8. Complete Pipelines

Run the entire ML pipeline from preprocessing to evaluation:

```bash
task full-pipeline
```

Or just run the data preparation steps:

```bash
task data-pipeline
```

## 9. Customizing Configuration

Edit the JSON configuration files in the `configs/` directory to customize:

- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings

## 10. Exploring Your Data

Launch the Jupyter notebook for interactive data exploration:

```bash
task explore
```

## Troubleshooting

If you encounter issues:

1. Check the logs for detailed error messages
2. Ensure your .mat files follow the expected structure
3. Verify the class mapping file is properly formatted
4. For memory issues during training, reduce batch size in the configs

For more detailed documentation, refer to the README.md file.