# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

**Student Name:** Muhammad Ibrahim Asghar  
**Registration Number:** FA23-BAI-046  
**Course:** Introduction to Computer Vision  

---

## 1. Introduction
Edge detection is a foundational low-level image processing operation designed to localize boundaries of objects by detecting sharp discontinuities in pixel intensity[cite: 2]. In medical image analysis, precisely isolating morphological boundaries is critical for characterizing lesion borders, geometry, and structural irregularity[cite: 2]. This laboratory systematically investigates first-order, second-order, and multi-stage edge detection operators—specifically Sobel ($G_x$, $G_y$, and magnitude), Prewitt, Laplacian, Laplacian of Gaussian (LoG), and Canny—applied across dermatological images from the ISIC skin cancer dataset[cite: 2]. 

Furthermore, this study examines detector vulnerability under artificial Gaussian and Salt-and-Pepper noise regimes, evaluates the restorative efficacy of Gaussian and Median spatial filters, and benchmarks classification performance across Raw, Filtered, and Edge-derived representations using both classical machine learning (SVM, Random Forest, KNN) and deep convolutional architectures (CNN Model 1 and CNN Model 2)[cite: 2].

---

## 2. Methodology
The experimental workflow is structured into three interdependent phases[cite: 2]:

1. **Comparative Edge Detection & Noise Sensitivity Analysis:** 
   Grayscale representative images from three distinct diagnostic classes (`melanoma`, `basal cell carcinoma`, `nevus`) are subjected to first-order derivative operators (Sobel and Prewitt) and second-order derivative operators (Laplacian and LoG)[cite: 2]. Next, controlled Gaussian noise ($\sigma = 25$) and Salt-and-Pepper noise ($d = 0.04$) are introduced to assess edge degradation[cite: 2]. Gaussian smoothing ($5 \times 5$, $\sigma = 0$) and Median filtering ($5 \times 5$) are subsequently applied prior to edge extraction to evaluate noise suppression and edge retention[cite: 2].

2. **Canny Multi-Stage Parameter Exploration:** 
   The Canny edge detector is parameterized across varying hysteresis thresholds ($T_{low} / T_{high} \in \{30/100, 50/150, 100/200\}$) and Gaussian pre-smoothing kernel dimensions ($3 \times 3$ vs. $5 \times 5$)[cite: 2]. Quantitative edge pixel counts and qualitative structural continuity are evaluated to identify the optimal configuration for skin lesion boundary preservation[cite: 2].

3. **Classification Pipeline Across Image Representations:** 
   Using the selected optimal Canny configuration and best filter, three distinct datasets are constructed from the ISIC dataset:
   * **Set A (Raw Images):** Original unenhanced grayscale inputs[cite: 2].
   * **Set B (Filtered Images):** Preprocessed images via $5 \times 5$ Gaussian filtering[cite: 2].
   * **Set C (Edge Images):** Binary edge maps generated using optimal Canny parameters[cite: 2].
   
   To guarantee statistical validity, identical sample indices are enforced across all three sets using stratified sampling (70% Training, 15% Validation, 15% Testing)[cite: 2]. Classifiers are trained under identical conditions to assess the diagnostic utility of boundary-only representations versus full-intensity spatial data[cite: 2].

---

## 3. Experimental Setup
* **Hardware & Acceleration:** Google Colab Environment accelerated by an NVIDIA Tesla T4 GPU (16 GB VRAM).
* **Dataset:** ISIC Skin Cancer dataset comprising 9 diagnostic classes (`actinic keratosis`, `basal cell carcinoma`, `dermatofibroma`, `melanoma`, `nevus`, `pigmented benign keratosis`, `seborrheic keratosis`, `squamous cell carcinoma`, `vascular lesion`).
* **Sampling & Resolution:** Input images were normalized to $128 \times 128$ spatial resolution and scaled to $[0, 1]$. To mitigate severe class imbalance and maintain computational tractability, a uniform cap of 100 samples per class was applied across the 9 classes.
* **Dataset Splitting:** Stratified partitioning across 9 classes yielded 630 training samples (70%), 135 validation samples (15%), and 135 test samples (15%). Identical sample splits were shared across Raw, Filtered, and Edge representations[cite: 2].
* **Hyperparameters:**
  * **Classical Models:** SVM ($RBF\text{ kernel}, C=1.0$), Random Forest ($n\_estimators=50$), KNN ($k=3$).
  * **CNN Model 1:** Shallow topology (Conv2D [16 filters, $3 \times 3$], MaxPooling [$2 \times 2$], Dense [32], Dropout [0.3], Softmax [9]).
  * **CNN Model 2:** Deeper topology (Conv2D [32 filters, $3 \times 3$], MaxPooling [$2 \times 2$], Conv2D [64 filters, $3 \times 3$], MaxPooling [$2 \times 2$], Dense [64], Dropout [0.4], Softmax [9]).
  * **Training Protocol:** Categorical cross-entropy loss, Adam optimizer ($\alpha = 0.001$), batch size of 32, evaluated over 5 epochs[cite: 2].

---

## 4. Results

### Table 1: Effect of Noise and Preprocessing on Edge Detection
*Visual inspection observations gathered from Task 2 evaluations[cite: 5].*

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Original | None | None | High | Moderate | Clear boundaries detected; thick edge profile[cite: 5]. |
| **Sobel** | Noisy | Gaussian | None | Poor | High | Many false edges due to background noise[cite: 5]. |
| **Sobel** | Noisy | Gaussian | Gaussian Filter | Good | Low | False edges reduced; structural edges slightly smoothed[cite: 5]. |
| **Sobel** | Noisy | Salt & Pepper | Median Filter | Good | Low | S&P noise completely isolated and removed prior to edge detection[cite: 5]. |
| **Prewitt** | Original | None | None | High | Moderate | Similar to Sobel but slightly less sensitive to diagonals[cite: 5]. |
| **Laplacian** | Original | None | None | High | Very High | Catches fine details but amplifies underlying skin texture[cite: 5]. |
| **LoG** | Noisy | Gaussian | Built-in smoothing | Good | Low | Internal Gaussian blur mitigates noise well[cite: 5]. |
| **Canny** | Original | None | None | Excellent | Low | Continuous, thin edges representing structural boundaries[cite: 5]. |

---

### Table 2: Canny Parameter Analysis
*Empirical evaluation on representative lesion image ($128 \times 128$)[cite: 6].*

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observation |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **Canny-1** | 30 | 100 | $3 \times 3$ | Over-segmented | 2,691 | High edge count; detects hair, skin pores, and noise artifacts[cite: 6]. |
| **Canny-2** | 50 | 150 | $3 \times 3$ | Balanced | 1,478 | Optimal lesion perimeter; thin, continuous, well-isolated boundaries[cite: 6]. |
| **Canny-3** | 100 | 200 | $3 \times 3$ | Under-segmented | 432 | Severe edge fragmentation; misses weak boundary transitions[cite: 6]. |
| **Canny-4** | 50 | 150 | $5 \times 5$ | Smoothed | 729 | Strong noise attenuation; however, perimeter localization is displaced[cite: 6]. |

*Selected Configuration:* **Canny-2** ($Low=50$, $High=150$, $Kernel=3 \times 3$) was selected as the optimal boundary descriptor for constructing Dataset C[cite: 2].

---

### Table 3: Cross-Lab Classification Performance Comparison
*Empirical test results ($N_{test} = 135$) across Raw, Filtered, and Edge representations[cite: 7].*

| Model / Classifier | Accuracy Raw | Accuracy Filtered | Accuracy Edge | Precision (Edge) | Recall (Edge) | F1-Score (Edge) | Training Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | 0.3130 | 0.3282 | 0.1985 | 0.2939 | 0.1985 | 0.1562 | 5.5609 | 1.5035 |
| **Random Forest** | 0.3588 | 0.3588 | 0.1985 | 0.2148 | 0.1985 | 0.1805 | 1.4086 | 0.0091 |
| **KNN** | 0.3511 | 0.3282 | 0.1069 | 0.0124 | 0.1069 | 0.0223 | 0.0044 | 0.1422 |
| **CNN Model 1** | 0.1145 | 0.1145 | 0.1985 | 0.1720 | 0.1985 | 0.1753 | 6.0568 | 1.3612 |
| **CNN Model 2** | 0.1145 | 0.2137 | 0.2214 | 0.1973 | 0.2214 | 0.1970 | 9.3596 | 0.6780 |

*Best Performing Model:* **CNN Model 2** achieved the highest accuracy on Edge representations (0.2214 / 22.14%)[cite: 7], while **Random Forest** achieved the highest overall accuracy across the experiment (0.3588 / 35.88% on Raw and Filtered sets)[cite: 7].

---

## 5. Discussion

### Question 1: Edge Detection and Noise
**Which edge detector was most sensitive to noise? Explain your answer using your experimental observations.**  
The Laplacian operator exhibited the highest sensitivity to noise[cite: 2, 5]. Mathematically, the Laplacian is an unoriented second-order differential operator ($\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$). Second derivatives are disproportionately sensitive to high-frequency fluctuations because differentiating a high-frequency component $\sin(\omega x)$ scales its amplitude by $\omega^2$. As shown in Table 1, applying the Laplacian to raw or noisy images detected numerous spurious zero-crossings, amplifying skin texture, hair, and noise into false edges[cite: 5]. Conversely, operators equipped with built-in smoothing (LoG and Canny) suppressed high-frequency noise prior to differentiation, maintaining boundary integrity[cite: 5].

### Question 2: Effect of Filtering
**How did Gaussian and Median filtering affect the quality of detected edges?**  
The filtering modality must match the statistical distribution of the noise[cite: 2]:
* **Gaussian Filtering:** Convolving with a $2D$ Gaussian distribution acts as a low-pass filter that effectively suppresses zero-mean Gaussian distributed perturbations by averaging local neighborhoods[cite: 2]. While this significantly curtailed false edges in Sobel and Prewitt outputs (improving edge quality from "Poor" to "Good"), it blurred rapid intensity transitions, marginally reducing edge sharpness and positional localization[cite: 5].
* **Median Filtering:** For impulsive Salt-and-Pepper noise, linear Gaussian filtering fails because extreme intensity outliers skew the weighted average. Median filtering replaces corrupted outlier pixels with local neighborhood medians, completely removing impulse spikes without blurring edge gradients[cite: 5]. This provided a clean input for subsequent gradient operators[cite: 5].

### Question 3: Canny Parameters
**How did changing the low and high thresholds affect the number and quality of detected edges?**  
Canny hysteresis utilizes dual thresholds ($T_{low}$ and $T_{high}$) to resolve edge fragmentation and noise streaking:
* **Lowering Thresholds (Canny-1: 30/100):** Produced 2,691 edge pixels[cite: 6]. The low $T_{low}$ permitted weak gradient responses to form connected edges, causing over-segmentation where benign epidermal textures and peripheral artifacts were registered as structural contours[cite: 6].
* **Elevating Thresholds (Canny-3: 100/200):** Produced 432 edge pixels[cite: 6]. Stringent gradient requirements suppressed minor noise but caused under-segmentation; true lesion perimeters with gradual diffuse transitions failed to exceed $T_{high}$, resulting in severely fractured and incomplete contours[cite: 6].
* **Kernel Dimension ($3 \times 3$ vs. $5 \times 5$):** Expanding the Gaussian pre-filter kernel from $3 \times 3$ (Canny-2) to $5 \times 5$ (Canny-4) at identical thresholds dropped edge pixel density from 1,478 to 729[cite: 6]. Larger smoothing kernels suppress fine gradient detail, yielding smoother boundary outlines at the expense of localized precision[cite: 6].

### Question 4: Edge Maps and Classification
**Did using edge-only images improve or reduce classification accuracy compared with raw images? Explain the possible reasons.**  
For classical machine learning classifiers, edge-only representations substantially reduced performance[cite: 2, 7]:
* Random Forest dropped from **35.88% (Raw)** to **19.85% (Edge)**[cite: 7].
* SVM dropped from **31.30% (Raw)** to **19.85% (Edge)**[cite: 7].
* KNN experienced a catastrophic drop from **35.11% (Raw)** to **10.69% (Edge)**[cite: 7].

This performance degradation occurs because classical classifiers relying on flattened spatial vectors depend heavily on dense intensity distributions, localized chromatic variance, and textural homogeneity. Reducing an image to a sparse binary edge map removes approximately 90% of the active pixel variance.

In contrast, CNN Model 1 and Model 2 performed poorly on Raw inputs (**11.45%**, equivalent to random chance across 9 classes) due to training a randomized network from scratch on a small sample size without data augmentation[cite: 7]. When fed Edge representations, CNN Model 2 improved to **22.14%**[cite: 7]. The binary edge map stripped away complex background illumination and varied skin tones, effectively simplifying the feature space into geometric silhouettes that a small CNN could separate more easily with limited data[cite: 7].

### Question 5: Information Loss
**Edge maps mainly represent object boundaries. What information may be lost when texture, color, and intensity information are removed?**  
In dermatological oncology, standard diagnostic frameworks (such as the ABCD rule: Asymmetry, Border irregularity, Color variegation, Diameter) depend directly on the features discarded during binarized edge extraction[cite: 2]:
* **Color Variegation:** Differences in melanin, hemoglobin, and collagen absorption (pigment variegation from tan to dark brown/black) are eliminated. Melanoma and benign nevi can share similar outer boundary geometries while exhibiting starkly different internal pigment patterns.
* **Internal Structural Texture:** Dermoscopic features like pigment networks, streaks, dots, and blue-white veiling exist as subtle intensity variations across the lesion surface rather than sharp step edges.
* **Contrast and Shading Gradients:** Edge binarization discards volumetric shading and transition gradations, treating all thresholded gradients as uniform binary boundaries.

### Question 6: Classical vs. Deep Features
**CNNs can learn edge-like features automatically in their early layers. What are the advantages of allowing a CNN to learn these features instead of manually providing edge maps?**  
Handcrafted edge detectors (such as Sobel or Canny) rely on fixed mathematical formulations (e.g., first or second spatial derivatives combined with static thresholding)[cite: 2]. They treat all gradients equally, regardless of whether a boundary represents a clinical lesion border or an irrelevant artifact (like a hair follicle or skin pore).

Conversely, the initial convolutional layers of a CNN initialize with unconstrained filter weights that optimize directly against the cross-entropy loss via backpropagation[cite: 2]. This enables the network to:
1. Learn task-specific oriented Gabor-like edge detectors sensitive to color transitions rather than merely scalar intensity steps[cite: 2].
2. Adapt spatial receptive fields dynamically across multiple scales[cite: 2].
3. Propagate learned edge representations into deeper hierarchical layers, assembling boundaries into texture motifs, parts, and complete lesion morphology[cite: 2]. Supplying hardcoded binary edge maps truncates this hierarchy, preventing the network from extracting higher-order semantic representations[cite: 2].

### Question 7: Best Representation
**Based on your results from Labs 01–03, which input representation produced the most useful classification results: Raw, Filtered, or Edge?**  
Based on the empirical results in Table 3, the **Filtered / Raw representations** provided the most useful inputs for reliable image classification[cite: 2, 7]:
* Classical models achieved their highest performance on **Raw and Filtered** images (**35.88%** accuracy via Random Forest)[cite: 7].
* Mild spatial filtering (Set B: Gaussian Filtered) matched or exceeded raw performance (SVM increased from **31.30%** to **32.82%**) by attenuating high-frequency acquisition noise while retaining spatial intensity distributions[cite: 7].
* The **Edge representation** resulted in significant performance drops across all classical classifiers (falling as low as **10.69%**)[cite: 7]. While CNN Model 2 achieved its highest accuracy on edge inputs (**22.14%** vs **11.45%** on Raw)[cite: 7], this reflects sample-constrained training dynamics rather than superior feature quality. Overall classification success relies on color and textural context, making Filtered and Raw images the superior representation[cite: 2].

---

## 6. Conclusion
This laboratory highlights the operational characteristics of standard edge detection techniques and demonstrates their downstream effects on pattern classification[cite: 2]. Unconstrained second-order operators (Laplacian) are highly vulnerable to noise amplification, whereas multi-stage hysteresis operators (Canny) deliver superior localized boundary delineation[cite: 2, 5]. 

The classification benchmarks demonstrate that removing color, intensity, and texture in favor of sparse edge maps leads to information loss that degrades classical classifiers (Random Forest, SVM, KNN)[cite: 2, 7]. While binary edge maps reduced input complexity for small CNNs trained from scratch, optimal overall accuracy required preserving dense spatial and chromatic features[cite: 2, 7]. Consequently, gentle spatial filtering that preserves textural context remains a more effective preprocessing strategy for complex diagnostic imagery than hard boundary extraction[cite: 2].

---

## 7. References
1. International Skin Imaging Collaboration (ISIC). (2026). *ISIC Archive: Skin Lesion Analysis Towards Melanoma Detection*.
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
3. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679–698.