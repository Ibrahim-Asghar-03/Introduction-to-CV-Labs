## Table 1. Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AlexNet | 50.85% | 52.96% | 50.69% | 47.97% | 87.66% |
| VGG16 | 54.24% | 54.81% | 53.47% | 49.71% | 88.06% |
| VGG19 | 53.39% | 56.75% | 52.78% | 49.39% | 87.78% |
| ResNet18 | 52.54% | 52.34% | 52.08% | 48.25% | 91.78% |
| ResNet50 | 55.93% | 59.34% | 54.86% | 51.52% | 91.27% |
| ResNet101 | 58.47% | 59.89% | 56.94% | 55.67% | 93.80% |
| DenseNet121 | 52.54% | 61.55% | 52.08% | 48.70% | 91.32% |
| EfficientNet-B0 | 62.71% | 59.21% | 60.42% | 55.68% | 91.75% |

<br>

## Table 2. Comparison of Different Classifiers

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Deep Features | Logistic Regression | 46.61% | 44.02% | 47.22% | 42.72% | 87.71% |
| Deep Features | Decision Tree | 36.44% | 30.76% | 35.88% | 30.63% | 63.86% |
| Deep Features | Random Forest | 38.14% | 36.04% | 40.28% | 30.98% | 80.45% |
| Deep Features | K-Nearest Neighbors (KNN) | 40.68% | 42.79% | 42.36% | 37.20% | 76.36% |
| Deep Features | Linear SVM | 43.22% | 41.88% | 44.44% | 39.24% | 89.16% |
| Deep Features | RBF-SVM | 47.46% | 51.85% | 47.92% | 42.92% | 90.05% |
| Deep Features | XGBoost | 44.92% | 45.24% | 45.83% | 39.81% | 79.57% |

<br>

## Table 3. Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AlexNet | 57.04M | 217.59MB | 1.42G | 2.45ms | 50.85% |
| VGG16 | 134.30M | 512.30MB | 31.04G | 11.46ms | 54.24% |
| VGG19 | 139.61M | 532.56MB | 39.37G | 14.95ms | 53.39% |
| ResNet18 | 11.18M | 42.65MB | 3.65G | 3.27ms | 52.54% |
| ResNet50 | 23.53M | 89.75MB | 8.26G | 8.33ms | 55.93% |
| DenseNet121 | 6.96M | 26.56MB | 5.80G | 14.79ms | 52.54% |
| EfficientNet-B0 | 4.02M | 15.33MB | 0.82G | 8.21ms | 62.71% |
