# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images
**Course:** Introduction to Computer Vision (Lab Mid Exam)  
**Student Name:** Muhammad Ibrahim Asghar  
**Registration Number:** FA23-BAI-046 (`046`)  
**Notebook File:** `FA23_BAI_046_Muhammad_Ibrahim_Asghar_LabMid.ipynb`

---

## Personal & Fixed Settings

### 1. Personal Settings (Derived from Registration Number `046`)
| Setting | Rule | My Value (Reg: `046`) |
| :--- | :--- | :--- |
| **Random Seed** | Last two digits of registration number | `46` |
| **HOG Cell Size** | Last digit even (`6` is even): $8 \times 8$. Last digit odd: $16 \times 16$ | $8 \times 8$ (`(8, 8)`) |
| **SIFT Grid Size** | Last digit 0 to 4: $4 \times 4$ grid (16 keypoints). Last digit 5 to 9 (`6` is in 5–9): $6 \times 6$ grid (36 keypoints) | $6 \times 6$ grid (`36 keypoints`) |

### 2. Fixed Settings for Everyone
| Fixed Setting | Value Used |
| :--- | :--- |
| **Dataset** | BloodMNIST |
| **Classes** | `eosinophil (1)`, `monocyte (5)`, `neutrophil (6)`, `platelet (7)` |
| **Images per Class** | `100` train, `30` validation, `30` test (sampled using `SEED = 46`) |
| **Image Preparation** | Grayscale, resized to $128 \times 128$ |
| **HOG Parameters** | `9` orientations, $2 \times 2$ cells per block (`L2-Hys` block normalization) |
| **Classifier** | `LinearSVC` (`random_state = 46`), $C \in \{0.1, 1, 10\}$ tuned on validation set |
| **Rotation Test** | $20^\circ$ counter-clockwise rotation applied before feature extraction |
| **Noise Test** | Gaussian noise with $\sigma = 0.05$ (image scaled to $[0, 1]$) applied before feature extraction |

---

## Q1. Data Setup (3 Marks)

### Implementation Summary
The BloodMNIST dataset was extracted locally into `/content/BloodMNIST` containing `['val', 'test', 'train']` splits. We filtered the dataset to retain only the four required target classes: `eosinophil (1)`, `monocyte (5)`, `neutrophil (6)`, and `platelet (7)`. Using `SEED = 46`, we randomly sampled `100` training, `30` validation, and `30` test images per class, converted every image to single-channel grayscale, and resized it to $128 \times 128$ pixels.

### Executed Output: Split Shapes & Class Counts
```text
Detected Dataset Root: /content/BloodMNIST
Subdirectories found: ['val', 'test', 'train']

=== SPLIT SHAPES & CLASS COUNTS ===
Train      | Shape: (400, 128, 128) | Class Counts: {'eosinophil (1)': 100, 'monocyte (5)': 100, 'neutrophil (6)': 100, 'platelet (7)': 100}
Val        | Shape: (120, 128, 128) | Class Counts: {'eosinophil (1)': 30, 'monocyte (5)': 30, 'neutrophil (6)': 30, 'platelet (7)': 30}
Test       | Shape: (120, 128, 128) | Class Counts: {'eosinophil (1)': 30, 'monocyte (5)': 30, 'neutrophil (6)': 30, 'platelet (7)': 30}
```

* **Visual Verification:** A $4 \times 4$ grid displaying 4 example grayscale images ($128 \times 128$) per class was generated and verified in the notebook output under Q1.

---

## Q2. HOG + SVM (6 Marks)

### (1) Hand Calculation of HOG Feature Vector Length
Given parameters for Registration Number **046**:
* **Image Size:** $128 \times 128$ pixels
* **HOG Cell Size:** $8 \times 8$ pixels (since last digit `6` is even)
* **Cells per Block:** $2 \times 2$ cells
* **Block Stride (Step):** $1 \times 1$ cell (default 50% block overlap)
* **Orientations:** $9$ bins

**Step-by-step calculation:**
1. **Grid of cells along width and height:**
   $$\text{Cells}_x = \frac{128}{8} = 16, \quad \text{Cells}_y = \frac{128}{8} = 16$$
2. **Number of $2 \times 2$ blocks across the $16 \times 16$ cell grid (with stride $= 1$):**
   $$\text{Blocks}_x = (16 - 2) + 1 = 15$$
   $$\text{Blocks}_y = (16 - 2) + 1 = 15$$
   $$\text{Total Blocks} = 15 \times 15 = 225 \text{ blocks}$$
3. **Features per block:**
   $$\text{Features per Block} = (2 \times 2 \text{ cells}) \times 9 \text{ orientations} = 4 \times 9 = 36$$
4. **Total HOG Feature Vector Length:**
   $$\text{Total HOG Features} = 225 \times 36 = \mathbf{8100}$$

**Executed Output Confirmation:**
```text
Extracted HOG Feature Vector Length per image: 8100
X_train_hog shape: (400, 8100) | X_val_hog shape: (120, 8100)
```

### (2) HOG Visualization
A $2 \times 4$ comparison figure showing one original grayscale blood-cell image (top row) alongside its corresponding $8 \times 8$ HOG gradient magnitude/orientation visualization (bottom row) for each of the four classes (`eosinophil`, `monocyte`, `neutrophil`, `platelet`) was generated and displayed in the notebook.

### (3) LinearSVC Training & Validation Evaluation (HOG)
We trained `LinearSVC` models on `X_train_hog` for $C \in \{0.1, 1, 10\}$ and evaluated them on `X_val_hog`:

```text
=== HOG + LinearSVC Hyperparameter Tuning on Validation Set ===
C = 0.1  | Val Accuracy: 74.17% (0.7417) | Val Macro-F1: 0.7396
C = 1    | Val Accuracy: 75.00% (0.7500) | Val Macro-F1: 0.7481
C = 10   | Val Accuracy: 75.00% (0.7500) | Val Macro-F1: 0.7484

>>> Best HOG Model Selected: C = 1
>>> HOG Validation Accuracy: 75.00% (0.7500)
>>> HOG Validation Macro-F1: 0.7481
```

#### HOG + SVM Validation Confusion Matrix ($C = 1$)
| True \ Predicted | eosinophil (1) | monocyte (5) | neutrophil (6) | platelet (7) | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **eosinophil (1)** | **20** | 5 | 5 | 0 | 30 |
| **monocyte (5)** | 4 | **21** | 3 | 2 | 30 |
| **neutrophil (6)** | 3 | 8 | **19** | 0 | 30 |
| **platelet (7)** | 0 | 0 | 0 | **30** | 30 |

---

## Q3. SIFT + SVM (7 Marks)

### (1) Normal SIFT Keypoint Detection (`detectAndCompute`)
Running standard DoG keypoint detection (`cv2.SIFT_create().detectAndCompute`) across all `400` training images produced the following statistics:

```text
=== Q3(1): Normal SIFT Detection Stats on Training Set (400 Images) ===
Average keypoints per image : 171.59
Minimum keypoints per image : 13
Maximum keypoints per image : 442
Images with zero keypoints  : 0
```
*(Rich keypoints were drawn and visualized on a sample training image with `190` detected keypoints).*

**Why a different number of keypoints per image is a problem for an SVM (One Sentence):**  
A standard Support Vector Machine (`LinearSVC`) requires every input image to be represented as a fixed-length feature vector $\mathbf{x} \in \mathbb{R}^d$ in a single consistent vector space so it can compute the linear decision boundary $\mathbf{w}^T \mathbf{x} + b$, which is impossible when `detectAndCompute` returns a variable number of $128$-D descriptors per image (ranging from $13 \times 128 = 1664$ up to $442 \times 128 = 56576$ features).

### (2) Fixed $6 \times 6$ Grid SIFT Feature Extraction & Hand Calculation
To guarantee a fixed-dimensional feature vector for every image, we placed keypoints on a regular $6 \times 6$ grid over the $128 \times 128$ image with keypoint diameter `size = 32` and computed descriptors via `sift.compute()`.

**Hand Calculation of Fixed-Grid SIFT Feature Length:**
* **SIFT Grid Size (Reg No `046`):** $6 \times 6 = 36$ grid keypoints (since last digit $6 \in [5, 9]$)
* **SIFT Descriptor Dimension per Keypoint:** $4 \times 4 \text{ spatial subregions} \times 8 \text{ orientation bins} = 128$
$$\text{Total SIFT Feature Length} = \text{Grid Points} \times 128 = 36 \times 128 = \mathbf{4608}$$

**Executed Output Confirmation:**
```text
=== Q3(2): Fixed 6x6 Grid SIFT Feature Extraction ===
Number of fixed grid keypoints : 36
Calculated SIFT Feature Length : 36 x 128 = 4608
X_train_sift shape: (400, 4608) | X_val_sift shape: (120, 4608)
```

### (3) LinearSVC Training & Validation Evaluation (SIFT)
We trained `LinearSVC` models on `X_train_sift` for $C \in \{0.1, 1, 10\}$ and evaluated them on `X_val_sift`:

```text
=== SIFT + LinearSVC Hyperparameter Tuning on Validation Set ===
C = 0.1  | Val Accuracy: 76.67% (0.7667) | Val Macro-F1: 0.7616
C = 1    | Val Accuracy: 77.50% (0.7750) | Val Macro-F1: 0.7717
C = 10   | Val Accuracy: 77.50% (0.7750) | Val Macro-F1: 0.7717

>>> Best SIFT Model Selected: C = 1
>>> SIFT Validation Accuracy: 77.50% (0.7750)
>>> SIFT Validation Macro-F1: 0.7717
```

#### SIFT + SVM Validation Confusion Matrix ($C = 1$)
| True \ Predicted | eosinophil (1) | monocyte (5) | neutrophil (6) | platelet (7) | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **eosinophil (1)** | **16** | 9 | 5 | 0 | 30 |
| **monocyte (5)** | 6 | **22** | 2 | 0 | 30 |
| **neutrophil (6)** | 3 | 2 | **25** | 0 | 30 |
| **platelet (7)** | 0 | 0 | 0 | **30** | 30 |

---

## Q4. Robustness (6 Marks)

### (1) Pre-Experiment Prediction (Written Before Running Test Evaluation)
* **Rotation ($20^\circ$):** We predicted that **HOG + SVM** would lose more accuracy under rotation because its fine $8 \times 8$ cell grid rigidly binds gradient histograms to exact spatial coordinates, whereas Dense SIFT pools gradients over larger $32 \times 32$ patches weighted by a smooth Gaussian window that tolerates minor spatial shifts.
* **Gaussian Noise ($\sigma = 0.05$):** We predicted that **HOG + SVM** would lose significantly more accuracy under Gaussian noise because HOG computes raw finite-difference pixel gradients ($[-1, 0, 1]$) directly on unsmoothed $8 \times 8$ cells—meaning high-frequency random pixel noise directly corrupts gradient magnitudes and directions—whereas SIFT applies Gaussian scale-space smoothing prior to derivative computation, effectively low-pass filtering high-frequency noise.

### (2) Test Set Evaluation Table (Frozen Models)
Both best models (`best_hog_model` with $C=1$ and `best_sift_model` with $C=1$) were frozen and evaluated on the `120` test images under three conditions (`Clean`, `Rotated by 20 degrees`, and `Noisy with Gaussian sigma = 0.05` applied before feature extraction):

| Model | Clean accuracy | Rotated accuracy ($20^\circ$) | Noisy accuracy ($\sigma = 0.05$) | Rotation Change | Noise Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HOG + SVM** | **71.67%** (`0.7167`) | **73.33%** (`0.7333`) | **62.50%** (`0.6250`) | `+1.67%` (Drop: `-1.67%`) | `-9.17%` (Drop: `+9.17%`) |
| **SIFT + SVM** | **67.50%** (`0.6750`) | **69.17%** (`0.6917`) | **72.50%** (`0.7250`) | `+1.67%` (Drop: `-1.67%`) | `+5.00%` (Drop: `-5.00%`) |

### (3) Was Your Prediction Correct? (Three-Sentence Explanation Using the Table)
My prediction for Gaussian noise was completely correct: **HOG + SVM** suffered a **9.17%** accuracy drop under noise (falling from **71.67%** clean to **62.50%** noisy), whereas **SIFT + SVM** lost no accuracy at all and actually improved by **5.00%** (rising from **67.50%** clean to **72.50%** noisy) because SIFT's Gaussian scale-space smoothing suppresses high-frequency pixel noise. However, my prediction for rotation was partially surprised by the domain geometry, as neither model lost accuracy under a $20^\circ$ rotation—both **HOG + SVM** (**71.67%** to **73.33%**) and **SIFT + SVM** (**67.50%** to **69.17%**) gained **1.67%** accuracy. This happened because microscopic blood-cell images are circular and naturally appear at arbitrary orientations in the training set, so rotating a test cell by $20^\circ$ simply aligns its nucleus lobes closer to rotated training examples.

---

## Q5. Short Answers and AI Log (3 Marks)

### (1a) Why does HOG use block normalisation? (2–3 Sentences)
HOG groups adjacent $8 \times 8$ cells into overlapping $2 \times 2$ blocks and normalizes their combined 36-D gradient histogram vector (using L2-Hys normalization) so that the descriptor becomes invariant to local illumination and contrast variations. When lighting or microscope staining intensity scales up or down across a region, all gradient magnitudes scale by the same factor, and dividing by the block norm cancels out this multiplicative contrast change while preserving relative edge structure.

### (1b) Why is it wrong to choose C using the test set? (2–3 Sentences)
Tuning the regularization hyperparameter $C$ on the test set causes data leakage (test-set overfitting), because the model's decision boundary is selected to favor the specific quirks and noise of the test samples. By choosing $C = 1$ strictly on the validation set (where HOG achieved **75.00%** and SIFT achieved **77.50%**) and touching the test set only once in Q4 with frozen models, our test accuracies remain an unbiased measure of real-world generalization.

### (2) Which two classes did your models confuse most, and why might grayscale make this harder?
In our validation confusion matrices, while `platelet` was classified with a perfect **30/30 (100%)** by both pipelines due to its small fragment size, the models most heavily confused **eosinophil vs. monocyte** (**15** total mutual errors in SIFT: **9** eosinophils predicted as monocyte + **6** monocytes predicted as eosinophil; and **9** mutual errors in HOG: **5** + **4**) as well as **monocyte vs. neutrophil** (**11** total mutual errors in HOG: **3** monocytes predicted as neutrophil + **8** neutrophils predicted as monocyte). Converting BloodMNIST images from RGB to **grayscale** makes distinguishing these white blood cells significantly harder because it strips away the diagnostic Wright-Giemsa staining colors—specifically the bright red-orange eosinophilic cytoplasmic granules of eosinophils versus the neutral pink/lilac cytoplasm of neutrophils and the bluish-gray cytoplasm of monocytes—forcing the descriptors to rely solely on grayscale luminance and nuclear shape.

### (3) AI Log
| My Prompt (One Line) | What I Changed | How I Checked the Result Was Correct |
| :--- | :--- | :--- |
| `"Build a Colab pipeline for BloodMNIST (classes 1, 5, 6, 7; 100/30/30 split) comparing HOG+LinearSVC and Dense-Grid SIFT+LinearSVC under 20-deg rotation and Gaussian noise (sigma=0.05)."` | Set personal parameters for Reg No `046` (`SEED=46`, `HOG_CELL_SIZE=(8,8)`, `SIFT_GRID_DIM=6`), modified the loader to unzip `BloodMNIST.zip` from Google Drive into `/content/BloodMNIST`, and converted noisy `[0,1]` float images back to `uint8` before calling `sift.compute()`. | Checked printed split shapes `(400, 128, 128)` and `(120, 128, 128)` with exact class counts (`100`/`30`/`30`), verified exact feature lengths (`8100` for HOG and `4608` for SIFT), and visually inspected the $4\times4$ class grid, HOG visualizations, SIFT keypoints (`190` on sample image), and confusion matrices. |