# MiniPlaces Image Classification with CNNs and ResNet

This project explores convolutional neural networks and modern deep learning techniques for scene classification on the MiniPlaces dataset. The notebook includes multiple model implementations, performance comparisons, and interpretability visualizations.

---



### 1. CNN from Scratch
Manual implementation of core convolution operations, including:
- Zero padding
- Stride based movement
- Multi channel kernel support
- Batched operations and correctness checks against `torch.nn.functional.conv2d`

### 2. FastCNN with PyTorch Modules
End-to-end trainable CNN using pooling, dropout, and ReLU activations to improve baseline performance

### 3. ResNet-18 Architecture
Custom residual blocks and full ResNet-18 recreated using PyTorch:
- Training from scratch
- Transfer learning with pretrained weights
- Linear probe vs end-to-end fine-tuning comparison

### 4. Class Activation Mapping (CAM)
Attention heatmaps that highlight where the model is “looking” when making predictions

### 5. Custom Model Challenge
Designed a deeper CNN architecture with channel scaling and stronger regularization:
- **69% accuracy**
- Ranked  **top 10** of submissions on the course Kaggle challenge  
  🔗 https://www.kaggle.com/competitions/cs-163-25-f-mini-places-classification-challenge
