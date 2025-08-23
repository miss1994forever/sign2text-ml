# Sign2Text ML Pipeline

This repository contains the machine learning pipeline for the Sign2Text project, which translates sign language videos to text using deep learning techniques.

## Project Structure

```
sign2text-ml/
├── ml/                         # Core ML components
│   ├── notebooks/              # Jupyter notebooks for data exploration
│   ├── data_processing/        # Scripts for data preparation
│   ├── models/                 # Neural network architecture definitions
│   ├── training/               # Training scripts
│   ├── evaluation/             # Model evaluation scripts
│   ├── requirements.txt        # Python dependencies
│   └── export_model.py         # Script to export models to CoreML format
├── data/                       # Raw data directory
├── processed_data/             # Will contain preprocessed data
├── models/                     # Will contain trained models
├── evaluation/                 # Will contain evaluation results
├── configs/                    # Configuration files
├── Taskfile.yml                # Task runner for the ML pipeline
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- [Task](https://taskfile.dev/) - Task runner (optional but recommended)
- CUDA-compatible GPU (optional, for faster training)

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sign2text-ml.git
   cd sign2text-ml
   ```

2. Install dependencies:
   ```bash
   # Using Task
   task install
   
   # Or with pip directly
   pip install -r ml/requirements.txt
   ```

### Quick Start

Run the complete ML pipeline with a single command:

```bash
task pipeline
```

This will:
1. Create necessary configuration files
2. Preprocess the sign language data
3. Split data into training/validation/test sets
4. Train the model
5. Evaluate the model performance

## Working with the Pipeline

### Available Tasks

The ML pipeline is broken down into modular tasks defined in `Taskfile.yml`:

- `task install`: Install all Python dependencies
- `task setup-dirs`: Create necessary directories
- `task preprocess`: Process raw sign language data
- `task split-data`: Split data into train/val/test sets
- `task train`: Train the CNN-LSTM model
- `task train-handpose`: Train the CNN-LSTM model with hand pose features
- `task evaluate`: Evaluate a trained model
- `task export`: Convert the model to CoreML format
- `task explore`: Open Jupyter notebook for data exploration
- `task create-configs`: Create configuration files

### Data Preprocessing

Process the sign language video data:

```bash
task preprocess
```

For the .mat files in your dataset:

```bash
task process-mat-files
```

### Training

Train a standard CNN-LSTM model:

```bash
task train
```

Or train with hand pose features:

```bash
task train-handpose
```

Customize training parameters by editing `configs/training_config.json`.

### Evaluation

Evaluate a trained model:

```bash
# Evaluate the latest model
task evaluate

# Evaluate a specific model
task evaluate MODEL_PATH=./models/my_model/model_best.pth MODEL_NAME=my_model
```

### Model Export

Export your trained model to CoreML format for iOS deployment:

```bash
task export MODEL_PATH=./models/my_model/model_best.pth
```

### Custom Pipeline

Run only specific steps of the pipeline:

```bash
task custom-pipeline TASK1=preprocess TASK2=split-data TASK3=train
```

## Model Architecture

The project implements two main architectures:

1. **CNN-LSTM**: A hybrid model that uses CNN for spatial feature extraction from video frames and LSTM for temporal processing of the frame sequence.

2. **CNN-LSTM with Hand Pose**: An enhanced model that incorporates MediaPipe hand landmarks as additional features.

Both models include attention mechanisms to focus on the most informative frames in the sequence.

## Data Format

The pipeline expects sign language data organized in the following format:
- Videos or image sequences in `data/` directory
- Each video representing a sign
- Labels indicating which sign is being performed

The preprocessor will extract frames and hand landmarks, saving them in a standardized format in the `processed_data/` directory.

## Citation

If you use this code in your research, please cite:

```
@misc{sign2text2023,
  author = {Your Name},
  title = {Sign2Text: Sign Language Translation System},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/sign2text-ml}
}
```

## License

[MIT License](LICENSE)