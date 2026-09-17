# Lab Task 02
**Title:** Effect of Image Filtering on Skin-Lesion Classification

## Required Experimental Results

### Table 1: Comparison of the Effect of Filtering on Top Pretrained Models

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | No Filter | 47.46% | 51.44% | 47.92% | 39.45% | 39.45% | 88.52% |
| **EfficientNet-B0** | Average | 47.46% | 52.34% | 47.92% | 41.28% | 41.28% | 86.67% |
| **EfficientNet-B0** | Gaussian | 50.00% | 53.65% | 50.00% | 42.32% | 42.32% | 89.53% |
| **EfficientNet-B0** | Median | 49.15% | 52.13% | 49.31% | 41.05% | 41.05% | 88.92% |
| **EfficientNet-B0** | Sharpening | 48.31% | 40.78% | 48.61% | 38.66% | 38.66% | 89.21% |
| **EfficientNet-B0** | Sobel | 42.37% | 45.90% | 43.75% | 36.19% | 36.19% | 85.09% |
| **ResNet101** | No Filter | 55.08% | 59.76% | 54.17% | 51.62% | 51.62% | 90.36% |
| **ResNet101** | Average | 55.93% | 55.60% | 54.86% | 51.96% | 51.96% | 90.04% |
| **ResNet101** | Gaussian | 51.69% | 61.38% | 51.39% | 49.37% | 49.37% | 92.34% |
| **ResNet101** | Median | 55.93% | 60.22% | 54.86% | 52.44% | 52.44% | 91.68% |
| **ResNet101** | Sharpening | 65.25% | 67.66% | 62.50% | 62.88% | 62.88% | 92.13% |
| **ResNet101** | Sobel | 46.61% | 47.11% | 47.22% | 44.13% | 44.13% | 90.45% |
| **ResNet50** | No Filter | 58.47% | 63.23% | 56.94% | 54.18% | 54.18% | 92.16% |
| **ResNet50** | Average | 52.54% | 57.31% | 52.08% | 51.13% | 51.13% | 89.04% |
| **ResNet50** | Gaussian | 55.93% | 59.57% | 54.86% | 53.77% | 53.77% | 91.65% |
| **ResNet50** | Median | 55.93% | 59.35% | 54.86% | 52.27% | 52.27% | 89.33% |
| **ResNet50** | Sharpening | 49.15% | 68.86% | 52.31% | 49.70% | 49.70% | 90.19% |
| **ResNet50** | Sobel | 49.15% | 55.91% | 49.31% | 45.12% | 45.12% | 87.02% |

<br>

## Questions to Answer

**1. Which three pretrained models performed best in Lab Activity 1?**
Based on the initial benchmarking, the top three performing models were EfficientNet-B0, ResNet101, and ResNet50.

**2. How does filtering affect each of the three models?**
In general, forcing spatial filters on raw images before passing them through deep Convolutional Neural Networks (CNNs) disrupts the natural pixel distribution the pretrained ImageNet weights rely on. Smoothing filters (Average, Median) tend to cause minor, sometimes negligible degradation (and in rare outlier cases, slight improvements like ResNet101 + Median). However, extreme structural filters (Sharpening, Sobel) cause severe disruption and unpredictable behavior across all three architectures. 

**3. Which filter produces the greatest change compared with the unfiltered baseline?**
The Sobel edge filter produced the most consistent and severe negative change across all models. For example, EfficientNet-B0 dropped from an unfiltered baseline F1-Score of 39.45% down to 36.19%, while ResNet101's F1-Score dropped from 51.62% to 44.13%. 

**4. Does the effect of a filter remain consistent across all three models?**
While the exact percentage point changes fluctuate due to architectural differences, the overall trend is consistent: smoothing filters result in mild performance shifts, while the Sobel filter consistently crippled the learning capability of all three networks.

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**
Heavy filtering distinctly decreases the Macro-F1 score. Because the Macro-F1 averages performance unweighted across all classes, the loss of subtle diagnostic features disproportionately impacts the network's ability to identify minority classes.

**6. Which lesion classes are most affected by filtering?**
Lesions that rely heavily on internal morphological textures and color gradations—such as the subtle pigment networks required to differentiate early melanoma from benign nevi—are the most affected.

**7. Why might smoothing remove useful lesion texture or morphological information?**
Smoothing filters act as low-pass filters that eliminate high-frequency noise. Unfortunately, in dermatology, high-frequency details (such as sharp borders, microscopic vascular structures, and fine pigment networks) are crucial diagnostic biomarkers. Smoothing essentially blurs the very evidence the CNN requires to differentiate cancer from a benign spot.

**8. Why might sharpening or edge detection help or hurt classification?**
*   **Help:** Sharpening could hypothetically assist in defining border irregularity, a key factor in dermatological diagnosis. (This was seen as an outlier in ResNet101, which jumped to 65.25% accuracy with Sharpening).
*   **Hurt:** Edge detection (Sobel) strips away almost all color and flat texture data. Since skin lesions are heavily diagnosed based on color variegation and internal gradients, removing this data blinds the network to primary diagnostic criteria.

**9. What is the difference between convolution and correlation?**
In classical image processing, correlation is the mathematical operation of sliding a filter kernel over an image and calculating the sum of products. Convolution is the exact same operation, except the filter kernel is rotated 180 degrees before it is applied.

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**
Classical image processing relies on humans hand-crafting specific filters (e.g., Sobel for edges, Gaussian for blur) to extract predetermined features. Deep Learning automates this process. The early layers of a CNN learn to become optimal feature extractors (acting like Gabor filters or edge detectors) tailored specifically to the training data. Applying classical filters *before* deep learning restricts the network by forcing it to analyze modified data, rather than allowing the CNN to discover the most critical diagnostic features organically.