# 🧠 Eregion - Neural Network Analytics & Monitoring

> **Real-time neural network monitoring and analytics for PyTorch and TensorFlow**

> ⚠️ **DEPRECATED** ⚠️
> 
> **This project is deprecated and no longer in active development.**
> 
> While the code may still be functional, we recommend using alternative neural network monitoring solutions for production use.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 What is Eregion?

Eregion is a powerful Python library that provides **real-time monitoring and analytics** for your neural networks. Whether you're training models with PyTorch or TensorFlow, Eregion gives you deep insights into your model's behavior, layer activations, gradients, and performance metrics.

### ✨ Key Features

- 🔍 **Layer-by-layer monitoring** - Track activations, gradients, and outputs from every layer
- 📊 **Advanced analytics** - Dead neuron detection, entropy analysis, gradient norms
- ☁️ **Cloud integration** - Push metrics to [eregion.dev](https://eregion.dev) for remote monitoring
- 🔄 **Auto-tracking** - Automatically monitor your model during training
- 🎯 **Framework agnostic** - Works seamlessly with PyTorch and TensorFlow
- 📈 **Real-time metrics** - Get instant feedback on model performance

## 🛠️ Installation

```bash
pip install eregion
```

Or install from source:

```bash
git clone https://github.com/BitLegion/eregion.git
cd eregion
pip install -e .
```

## 🚀 Quick Start

### PyTorch Example

```python
import torch
import torch.nn as nn
from eregion import EregionPyTorch

# Your model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Initialize Eregion monitoring
tracker = EregionPyTorch(
    model=model,
    name="my_pytorch_model",
    API_KEY="your_api_key_here",
    auto_track=True  # Automatically track all layers
)

# Train your model
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        # Push metrics to cloud
        tracker.push({"loss": loss.item(), "epoch": epoch})
```

### TensorFlow Example

```python
import tensorflow as tf
from eregion import EregionTensorFlow

# Your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Initialize Eregion monitoring
tracker = EregionTensorFlow(
    model=model,
    name="my_tensorflow_model",
    API_KEY="your_api_key_here",
    auto_track=True
)

# Train your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

for epoch in range(10):
    history = model.fit(
        train_dataset,
        epochs=1,
        verbose=0
    )
    
    # Push metrics to cloud
    tracker.push({
        "loss": history.history['loss'][0],
        "epoch": epoch
    })
```

## 📊 Analytics Features

### Layer Monitoring
- **Activation tracking** - Monitor output distributions from each layer
- **Gradient analysis** - Track gradient norms and vanishing/exploding gradients
- **Shape monitoring** - Automatic detection of tensor shapes and dimensions

### Advanced Metrics
- **Dead neuron detection** - Identify neurons with zero or near-zero activations
- **Entropy analysis** - Measure information content in layer outputs
- **Activation distribution** - Analyze statistical properties of activations

### Cloud Integration
- **Real-time dashboard** - View metrics on [eregion.dev](https://eregion.dev)
- **Historical tracking** - Compare performance across training runs
- **Team collaboration** - Share model insights with your team

## 🔧 Configuration

### API Setup
1. Sign up at [eregion.dev](https://eregion.dev)
2. Get your API key from the dashboard
3. Use the key in your Eregion tracker initialization

### Auto-tracking Options
```python
# Enable automatic layer monitoring
tracker = EregionPyTorch(
    model=model,
    name="my_model",
    API_KEY="your_key",
    auto_track=True,  # Automatically monitor all layers
    reset=True        # Reset existing model data
)
```

### Manual Data Pushing
```python
# Push custom metrics
tracker.push({
    "custom_metric": 0.95,
    "training_step": 1000,
    "learning_rate": 0.001
})

# Push layer-specific data
tracker.push({
    "layer_name": "conv1",
    "activation_mean": 0.1,
    "activation_std": 0.05
})
```

## 📈 What You Can Monitor

### Training Metrics
- Loss values and convergence
- Learning rate schedules
- Gradient norms and distributions
- Parameter updates

### Model Health
- Dead neurons and saturation
- Layer activation patterns
- Weight distributions
- Model complexity metrics

### Performance Insights
- Training vs validation metrics
- Overfitting detection
- Model efficiency analysis
- Resource utilization

### Development Setup
```bash
git clone https://github.com/pratyaypandey/eregion.git
cd eregion
pip install -e ".[dev]"
pytest tests/
```
