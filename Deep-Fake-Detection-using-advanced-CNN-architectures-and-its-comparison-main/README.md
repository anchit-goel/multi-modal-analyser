# DeepFake Detection Using customized CNNs (with optimizations)

# Project Overview
This project focuses on developing a DeepFake detection system using Convolutional Neural Networks (CNNs) and other deep learning models. The goal is to identify AI-generated or manipulated facial images and videos, which are increasingly used in misinformation, identity theft, and malicious activities.

## Dataset Description

The dataset was created by combining two publicly available datasets from Kaggle:

- **140K Real and Fake Faces**
- **Real and Fake Face Detection**

This resulted in a balanced dataset consisting of 142,356 images (71,178 real and 71,178 fake). All images were resized to 224×224 pixels. Preprocessing steps included:

- Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
- Data augmentation (random shear, horizontal flips, and rotations).
- The dataset was divided as follows:
  - 80% training
  - 10% validation
  - 10% testing

Relevant sources and discussions:
- Real Faces Dataset: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/122786  
- 1 Million Fake Faces Dataset: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121173

# Performance Evaluation:

## Performance Evaluation

| Model         | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------|----------|-----------|--------|----------|---------|
| VGGFace16     | 95.2%    | 96.0%     | 94.0%  | 94.8%    | 0.9601  |
| DenseNet-121  | 96.8%    | 97.2%     | 96.0%  | 96.6%    | 0.9924  |
| Custom CNN    | 95.0%    | 95.5%     | 94.5%  | 95.0%    | 0.9956  |

# Key Components

Models Used:

- custom_cnn.ipynb: A custom-built CNN architecture for detecting DeepFakes.
- densenet.ipynb: Implementation of DenseNet for feature extraction and classification.
- vggface.ipynb: Utilizes VGGFace for face recognition and forgery detection.
- DeepFake_hybrid_model_4_epochs.ipynb: A hybrid model trained over 4 epochs, combining multiple architectures.
- XAI_implementation.ipynb: Incorporates Explainable AI (XAI) techniques to interpret model predictions.
- XAI_implementation_finalized_4epoch_3-3.ipynb: Finalized version with optimized XAI visualization for better interpretability.



## Explainability and Interpretability

To improve transparency and trust in model predictions, the following XAI techniques were implemented:

### SHAP (SHapley Additive Explanations)
- Highlights regions of the face that positively or negatively contribute to model decisions.

### LIME (Local Interpretable Model-agnostic Explanations)
- Provides interpretable local approximations around individual predictions.

### Saliency, LRP, and Integrated Gradients
- These techniques helped visualize spatial attention across facial regions and confirmed that critical features were consistently utilized during prediction.

# Report:

RP_report_draft_DeepFake_detection.pdf: Draft of the research paper, covering methodology, results, and analysis.(to be made available soon with updations)

⚙️# Technologies and Tools

- Frameworks: TensorFlow, Keras, and PyTorch.
- Techniques: Transfer learning, hybrid models, and XAI visualization.
- Evaluation Metrics: Accuracy, precision, recall, and F1-score.
  
📊 # Project Goals:
- Develop a robust DeepFake detection model using CNNs.
- Enhance interpretability with Explainable AI techniques.
- Achieve high accuracy and low false-positive rates.

Remaining goals in progress: Deployment through Web applications(Hugging Face)
