# LAB 05: HOG-Based Industrial Defect Detection and Classification

## 1. Introduction
The automation of quality control is a critical component of modern manufacturing. This laboratory project explores the application of computer vision, specifically the Histogram of Oriented Gradients (HOG) feature descriptor combined with machine learning classifiers, to identify and classify manufacturing defects. The primary objective is to develop a robust prototype capable of automated visual inspection of industrial surfaces. By extracting shape, edge, and gradient information via HOG, this system determines whether a product is defective and categorizes the specific type of defect present.

## 2. Industrial Application
In traditional manufacturing environments, manual inspection of products is highly time-consuming, expensive, and susceptible to human error caused by visual fatigue or subjective judgment. An automated visual inspection system mitigates these issues by providing consistent, high-speed, and objective quality assessments. The deployment of a computer vision system ensures that defective products (e.g., those with surface scratches or structural weakness) are rejected before reaching the consumer, thereby maintaining brand reputation, reducing material waste, and optimizing the production pipeline.

## 3. Dataset Description
The model was trained and evaluated using the NEU Steel Surface Defect Database (NEU-DET). Moving beyond simple binary classification, this project addressed the advanced requirement of multi-class defect classification. The dataset encompasses six distinct defect classes:
*   `crazing`
*   `inclusion`
*   `patches`
*   `pitted_surface`
*   `rolled-in_scale`
*   `scratches`

The data was partitioned into a training set containing 1,700 images and a validation set containing 100 images. This split ensures a substantial amount of data for feature extraction while reserving a completely unseen subset for unbiased performance evaluation.

## 4. Image Preprocessing
To standardize the input pipeline and optimize computational efficiency, several preprocessing steps were applied to the raw dataset:
1.  **Resizing:** All images were resized to a common resolution of 64x64 pixels. This unified dimension accelerates HOG feature extraction without severely compromising the macro-level defect structures.
2.  **Grayscale Conversion:** The images were converted to single-channel grayscale. This is a critical step, as the HOG algorithm relies on pixel intensity gradients (edges and shapes) rather than color information to define features.

## 5. HOG Feature Extraction
The Histogram of Oriented Gradients (HOG) is a feature descriptor utilized extensively for object detection. It operates by counting occurrences of gradient orientation in localized portions of an image. By dividing the image into small spatial regions (cells) and compiling a 1D histogram of gradient directions for the pixels within each cell, HOG effectively captures the contour and texture of surface defects. 
Visualizations generated during the experiment confirmed that the HOG representation successfully isolated the distinct edges and textural anomalies inherent to different defect types, abstracting the raw grayscale image into a highly informative feature vector.

## 6. Classification Methodology
To evaluate the efficacy of the HOG features, two distinct machine learning classifiers were trained and compared:
*   **Support Vector Machine (SVM):** A linear SVM was used as the baseline classifier. SVMs are traditionally paired with HOG features (historically notable in pedestrian detection) because they perform well in high-dimensional feature spaces.
*   **XGBoost Classifier:** As the required additional classifier, Extreme Gradient Boosting (XGBoost) was employed. This tree-based ensemble method is highly robust against overfitting and capable of modeling complex, non-linear relationships within the feature set.

## 7. Experimental Setup
The system pipeline was developed in Python using Google Colab. Key libraries included `OpenCV` and `scikit-image` for image processing and HOG extraction, alongside `scikit-learn` and `xgboost` for model training and evaluation. 
A comprehensive hyperparameter experiment was conducted to investigate the effect of different HOG parameters. The tested variables included:
*   **Cell Sizes:** $4\times4$, $8\times8$, and $16\times16$
*   **Orientations:** 6, 9, and 12

Models were subsequently evaluated under varying realistic environmental conditions to assess robustness.

## 8. Results
The comparative analysis between the SVM and XGBoost classifiers demonstrated clear differences in predictive capability. 

**Classifier Comparison:**
| Metric | SVM | XGBoost |
| :--- | :--- | :--- |
| **Accuracy** | 0.7600 | **0.8200** |
| **Precision (Weighted)** | 0.7734 | **0.8249** |
| **Recall (Weighted)** | 0.7600 | **0.8200** |
| **F1-Score (Weighted)**| 0.7532 | **0.8200** |

XGBoost outperformed the Linear SVM across all metrics, achieving an overall accuracy of 82%. Analysis of the confusion matrices revealed that XGBoost was notably superior at distinguishing `inclusion` (0.94 recall vs SVM's 0.82). The SVM struggled more with `pitted_surface`, frequently confusing it with `patches` and `inclusion`.

**HOG Parameter Experiment Findings:**
The parameter search revealed that larger cell sizes combined with moderate orientations yielded the best results. 
*   **Best Configuration:** $16\times16$ cell size with 9 orientations achieved the highest accuracy (0.87) and F1-Score (0.8623).
*   **Worst Configuration:** $4\times4$ cell size with 9 orientations resulted in the lowest accuracy (0.69) and F1-Score (0.6864). 
This indicates that overly granular cell sizes capture too much noise in the steel textures, while larger cells successfully capture the broader structural shape of the defects.

## 9. Robustness Analysis
An industrial environment is rarely visually perfect. The system (using the baseline XGBoost model) was evaluated under three simulated imaging degradations:

| Condition | Accuracy | F1-Score | Accuracy Drop | F1 Drop |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.8200 | 0.8200 | 0.00 | 0.0000 |
| **Brightness (+50%)** | 0.6000 | 0.5935 | 0.22 | 0.2265 |
| **Gaussian Noise** | 0.2300 | 0.1452 | 0.59 | 0.6748 |
| **Blur (Sigma=2)** | 0.2000 | 0.0975 | 0.62 | 0.7225 |

The system is highly vulnerable to Blur and Gaussian Noise. Because HOG relies entirely on calculating image gradients (edges), blurring the image destroys gradient magnitude, effectively blinding the model. Noise introduces artificial micro-gradients, confusing the histograms. The model showed moderate resilience to brightness changes, which aligns with HOG's inherent partial invariance to uniform illumination variations.

## 10. Industrial Deployment Discussion
To deploy this module on a real factory floor, it must be integrated with a programmable logic controller (PLC) and an optical rig. The final output module is already structured to trigger mechanical actions, as demonstrated by the output format:
```text
PRODUCT INSPECTION RESULT
Prediction: DEFECTIVE (crazing)
Confidence: 92.45%
Action: REJECT PRODUCT
```
Given the findings from the robustness analysis, successful deployment absolutely requires a controlled optical environment. The camera rig must have an enclosure to prevent external brightness fluctuations, high-frequency lighting to prevent sensor noise, and strict focal constraints to prevent blurring. 

## 11. Limitations
While computationally lightweight, the primary limitation of this HOG-based approach is its dependency on manual feature engineering. It relies heavily on strict optical conditions (as proven by the blur/noise failure rates). It lacks the spatial invariance found in modern Deep Learning architectures. A Convolutional Neural Network (CNN) would likely extract more robust, hierarchical features that are resilient to noise and blur, albeit at a higher computational cost.

## 12. Conclusion
This laboratory successfully implemented an automated visual inspection prototype for industrial quality control using HOG features. By evaluating various classifiers, it was determined that an XGBoost classifier paired with $16\times16$ cell size HOG features provided the most accurate multi-class defect detection (87% peak accuracy). While the system proved highly effective on clean data and successfully integrated an automated accept/reject decision module, the robustness analysis highlighted the critical necessity for controlled factory lighting and strict camera focus to ensure reliable real-world deployment.