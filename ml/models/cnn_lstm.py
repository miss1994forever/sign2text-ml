#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN-LSTM model for sign language recognition.

This module implements a hybrid CNN-LSTM model for sign language recognition,
where CNN extracts spatial features from frames and LSTM captures temporal relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union, List


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for sign language recognition.

    The model consists of:
    1. CNN backbone for feature extraction from video frames
    2. LSTM to process temporal sequences of CNN features
    3. Attention mechanism to focus on important frames
    4. Classification head for sign prediction
    """

    def __init__(
        self,
        num_classes: int,
        frame_dim: Tuple[int, int, int] = (3, 224, 224),
        cnn_out_dim: int = 512,
        lstm_hidden_dim: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        attention: bool = True
    ):
        """
        Initialize the CNN-LSTM model.

        Args:
            num_classes: Number of sign classes to predict
            frame_dim: Dimensions of input frames (channels, height, width)
            cnn_out_dim: Dimension of CNN output features
            lstm_hidden_dim: Hidden dimension of LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            attention: Whether to use attention mechanism
        """
        super(CNNLSTM, self).__init__()

        # Store parameters
        self.num_classes = num_classes
        self.frame_dim = frame_dim
        self.cnn_out_dim = cnn_out_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.use_attention = attention

        # CNN for feature extraction
        self.cnn = self._build_cnn_backbone()

        # LSTM for sequence processing
        self.lstm_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        if self.use_attention:
            self.attention = nn.Linear(
                lstm_hidden_dim * self.lstm_directions, 1
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim * self.lstm_directions, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _build_cnn_backbone(self) -> nn.Module:
        """
        Build CNN backbone for feature extraction.

        Returns:
            CNN module for feature extraction
        """
        # Use a simplified CNN architecture
        # In practice, you might want to use a pre-trained model like ResNet
        cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(self.frame_dim[0], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),

            # Flatten to feature vector
            nn.Flatten()
        )

        return cnn

    def _initialize_weights(self):
        """Initialize weights for the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, channels, height, width)
                representing a batch of video sequences

        Returns:
            Dictionary containing:
                - 'logits': Logits for class prediction
                - 'attention_weights': Attention weights if attention is used
        """
        batch_size, seq_len, c, h, w = x.size()

        # Process each frame with CNN
        # Reshape to (batch_size * seq_len, c, h, w)
        x_reshaped = x.view(-1, c, h, w)
        cnn_features = self.cnn(x_reshaped)

        # Reshape back to (batch_size, seq_len, cnn_out_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # Process sequence with LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        # lstm_out shape: (batch_size, seq_len, lstm_hidden_dim * directions)

        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1)

            # Apply attention weights
            context = torch.bmm(
                attention_weights.unsqueeze(1),
                lstm_out
            ).squeeze(1)

            # Get predictions
            logits = self.classifier(context)

            return {
                'logits': logits,
                'attention_weights': attention_weights
            }
        else:
            # Use last hidden state for prediction if no attention
            if self.bidirectional:
                # Concatenate last hidden state from both directions
                hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden_cat = hidden[-1]

            # Get predictions
            logits = self.classifier(hidden_cat)

            return {
                'logits': logits
            }


class CNNLSTMWithHandPose(CNNLSTM):
    """
    Extended CNN-LSTM model that incorporates hand pose features from MediaPipe.

    This model takes both video frames and hand landmarks as inputs.
    """

    def __init__(
        self,
        num_classes: int,
        frame_dim: Tuple[int, int, int] = (3, 224, 224),
        landmark_dim: int = 21*3*2,  # 21 points per hand × 3D × 2 hands
        cnn_out_dim: int = 512,
        lstm_hidden_dim: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        attention: bool = True
    ):
        """
        Initialize the CNN-LSTM model with hand pose input.

        Args:
            num_classes: Number of sign classes to predict
            frame_dim: Dimensions of input frames (channels, height, width)
            landmark_dim: Dimension of hand landmarks feature vector
            cnn_out_dim: Dimension of CNN output features
            lstm_hidden_dim: Hidden dimension of LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            attention: Whether to use attention mechanism
        """
        super(CNNLSTMWithHandPose, self).__init__(
            num_classes, frame_dim, cnn_out_dim, lstm_hidden_dim,
            lstm_layers, dropout, bidirectional, attention
        )

        # Store landmark dimension
        self.landmark_dim = landmark_dim

        # Landmark processing network
        self.landmark_net = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 256, cnn_out_dim),
            nn.ReLU()
        )

    def forward(
        self,
        frames: torch.Tensor,
        landmarks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            frames: Input tensor of shape (batch_size, seq_len, channels, height, width)
                   representing a batch of video sequences
            landmarks: Input tensor of shape (batch_size, seq_len, landmark_dim)
                       representing hand landmarks

        Returns:
            Dictionary containing:
                - 'logits': Logits for class prediction
                - 'attention_weights': Attention weights if attention is used
        """
        batch_size, seq_len, c, h, w = frames.size()

        # Process each frame with CNN
        frames_reshaped = frames.view(-1, c, h, w)
        cnn_features = self.cnn(frames_reshaped)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # Process landmarks
        landmarks_reshaped = landmarks.view(-1, self.landmark_dim)
        landmark_features = self.landmark_net(landmarks_reshaped)
        landmark_features = landmark_features.view(batch_size, seq_len, -1)

        # Fuse features
        combined_features = torch.cat([cnn_features, landmark_features], dim=2)
        fused_features = self.fusion(combined_features)

        # Process sequence with LSTM
        lstm_out, (hidden, cell) = self.lstm(fused_features)

        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1)

            # Apply attention weights
            context = torch.bmm(
                attention_weights.unsqueeze(1),
                lstm_out
            ).squeeze(1)

            # Get predictions
            logits = self.classifier(context)

            return {
                'logits': logits,
                'attention_weights': attention_weights
            }
        else:
            # Use last hidden state for prediction if no attention
            if self.bidirectional:
                # Concatenate last hidden state from both directions
                hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden_cat = hidden[-1]

            # Get predictions
            logits = self.classifier(hidden_cat)

            return {
                'logits': logits
            }


def create_model(
    model_name: str,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_name: Name of the model to create
        num_classes: Number of classes for classification
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized model
    """
    models = {
        'cnn_lstm': CNNLSTM,
        'cnn_lstm_handpose': CNNLSTMWithHandPose,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")

    return models[model_name](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test the models
    batch_size = 4
    seq_len = 30
    frame_dim = (3, 224, 224)
    landmark_dim = 21*3*2  # 21 landmarks per hand, 3D coordinates, 2 hands
    num_classes = 100

    # Test base CNN-LSTM
    model = create_model('cnn_lstm', num_classes, frame_dim=frame_dim)
    frames = torch.randn(batch_size, seq_len, *frame_dim)
    output = model(frames)

    print(f"CNN-LSTM Output: {output['logits'].shape}")

    # Test CNN-LSTM with hand pose
    model_with_hands = create_model('cnn_lstm_handpose', num_classes, frame_dim=frame_dim, landmark_dim=landmark_dim)
    frames = torch.randn(batch_size, seq_len, *frame_dim)
    landmarks = torch.randn(batch_size, seq_len, landmark_dim)
    output = model_with_hands(frames, landmarks)

    print(f"CNN-LSTM with HandPose Output: {output['logits'].shape}")
