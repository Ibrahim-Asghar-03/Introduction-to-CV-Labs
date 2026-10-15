# Landmark Recognition & Localization Using the SIFT Algorithm

**Course:** Introduction to Computer Vision (Lab Assignment 02 / Lab 06)  
**Author:** Muhammad Ibrahim Asghar  
**Registration No:** FA23-BAI-046  
**Institution:** COMSATS University Islamabad, Wah Campus  

---

## 1. Executive Summary & System Architecture

This report presents the experimental evaluation of an automated **Landmark Recognition and Geometric Localization System** built using Python and OpenCV. Designed for a practical tourism information center use case, the pipeline identifies famous world landmarks from user-uploaded query photographs despite substantial real-world variations in scale, in-plane rotation, illumination, and 3D viewpoint.

### Core Pipeline Stages
1. **Reference Knowledge Base Construction (Task 1 & Task 2):** Multi-angle reference images across five landmark categories are standardized (`MAX_DIM = 600`), converted to grayscale, and processed using the **Scale-Invariant Feature Transform (`cv2.SIFT_create(nfeatures=500)`)** to extract scale- and rotation-invariant keypoints alongside 128-dimensional gradient histogram descriptors.
2. **Query Image Feature Extraction (Task 3):** Incoming query images are preprocessed and decomposed into SIFT keypoints and $N \times 128$ descriptor matrices.
3. **Brute-Force KNN Matching & Lowe's Ratio Test (Task 4):** Query descriptors are matched against the reference database using a **Brute-Force Matcher (`cv2.BFMatcher`)** with the **Euclidean distance ($L_2$ norm)** metric and **K-Nearest-Neighbor ($k = 2$)** search. Ambiguous matches are filtered via **Lowe's Ratio Test** ($d_1 < 0.75 \times d_2$), and candidate landmarks are ranked by their maximum single-image reliable matches and total class matches.
4. **RANSAC Geometric Verification & Localization (Task 5):** Point correspondences between the top-matching reference image and query image are passed to **Random Sample Consensus (`cv2.findHomography` with `cv2.RANSAC`, reprojection threshold = `5.0` px)**. When a valid, convex homography transformation $H$ with sufficient inliers ($\ge 7$) is found, the four corners of the reference image are projected onto the query image via `cv2.perspectiveTransform` to bound and localize the recognized landmark.
5. **Comprehensive Multi-Condition Evaluation (Task 6):** The end-to-end system is benchmarked across **339 query images** spanning five known landmark classes and an unassociated `unknown` test suite.

---

## 2. Reference Knowledge Base Statistics (Task 2)

The reference database was constructed from **1,322 images** across five famous landmarks in **56.32 seconds**, yielding a searchable knowledge base of **540,407 SIFT descriptors** (128-D vectors):

| Landmark Category | Reference Images | Total Extracted Keypoints | Avg. Keypoints / Image | Descriptor Tensor Shape |
| :--- | :---: | :---: | :---: | :---: |
| `christ_the_reedemer` | 258 | 85,682 | 332.1 | `(85682, 128)` |
| `eiffel_tower` | 312 | 118,725 | 380.5 | `(118725, 128)` |
| `great_wall_of_china` | 313 | 129,503 | 413.7 | `(129503, 128)` |
| `roman_colosseum` | 315 | 152,652 | 484.6 | `(152652, 128)` |
| `taj_mahal` | 126 | 53,845 | 427.3 | `(53845, 128)` |
| **Total / Overall** | **1,322** | **540,407** | **408.8** | **`(540407, 128)`** |

* **Observation:** Highly textured architectural structures such as the **Roman Colosseum** (`484.6` keypoints/image) and **Taj Mahal** (`427.3` keypoints/image) consistently saturate the 500-keypoint cap, whereas **Christ the Redeemer** (`332.1` keypoints/image) exhibits fewer keypoints due to large uniform sky backgrounds surrounding the statue.

---

## 3. Query Processing, Feature Matching & Localization Demos (Tasks 3–5)

### 3.1 Task 3: Query Feature Extraction
* **Sample Query Image:** `00d2585f75.jpg` (`christ_the_reedemer`)
* **Detected SIFT Keypoints:** `501`
* **Extracted Descriptor Matrix Shape:** `(501, 128)` (Keypoints $\times$ 128-D SIFT Vector)
* **Spatial Distribution:** Rich SIFT keypoints accurately cluster along high-contrast structural edges, the statue's vertical folds, pedestal geometry, and foreground visitors while ignoring uniform sky gradients.

### 3.2 Task 4: Candidate Ranking & Ratio Test Filtering

#### Test 1: Known Landmark Query (`00d2585f75.jpg` — Ground Truth: `christ_the_reedemer`)
* **Best Matching Reference Image:** `d3eec296fd.jpg` (`90` reliable matches passing Lowe's ratio test at $\tau = 0.75$)
* **Predicted Landmark:** `christ_the_reedemer` (**Status:** `Recognized`)

| Rank | Landmark Candidate | Best Single-Image Matches | Total Class Matches |
| :---: | :--- | :---: | :---: |
| **1** | **`christ_the_reedemer`** | **90** | **1,916** |
| 2 | `great_wall_of_china` | 50 | 1,401 |
| 3 | `eiffel_tower` | 33 | 1,507 |
| 4 | `taj_mahal` | 25 | 688 |
| 5 | `roman_colosseum` | 16 | 1,383 |

#### Test 2: Unknown Landmark Query (`0016510b5c.jpg` — Ground Truth: `unknown` / Statue of Liberty)
* **Raw Feature Matching Behavior:** Produced `151` raw descriptor matches against a low-contrast foggy reference image in `great_wall_of_china`.
* **Geometric Verification Result (Task 5):** **0 RANSAC Inliers** (`Unreliable Geometric Transformation (Homography Failed)`), confirming that although raw $L_2$ descriptor matching can be fooled by repetitive low-contrast textures, RANSAC geometric verification successfully identifies the match as spatially incoherent.

| Rank | Landmark Candidate | Best Single-Image Matches | Total Class Matches | RANSAC Inliers |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `great_wall_of_china` | 151 | 4,530 | **0 (Homography Failed)** |
| 2 | `eiffel_tower` | 56 | 4,822 | 0 |
| 3 | `christ_the_reedemer` | 56 | 4,537 | 0 |
| 4 | `taj_mahal` | 53 | 2,157 | 0 |
| 5 | `roman_colosseum` | 50 | 4,711 | 0 |

### 3.3 Task 5: Geometric Localization Verification
* **Known Query (`00d2585f75.jpg`):** Achieved **90 Good Matches** and **87 RANSAC Inliers** (**96.7% inlier ratio**) in `7.058s`, projecting a crisp green convex bounding polygon around Christ the Redeemer.
* **Unknown Query (`0016510b5c.jpg`):** Produced **151 Good Matches** clustered into just two degenerate point hubs on a foggy reference image, resulting in **0 RANSAC Inliers** (`Homography Failed`) and preventing an invalid bounding box from being drawn.

---

## 4. Comprehensive Performance Evaluation (Task 6)

### 4.1 Overall System Metrics
The complete evaluation suite of **339 query images** was benchmarked against all **1,322 reference images** (448,158 pairwise image comparisons):

* **Total Query Images Evaluated:** `339`
* **Correct Recognitions / Rejections:** `215 / 339`
* **Overall System Accuracy:** `63.42%`
* **Average Processing Time per Query Image:** `6.0230 seconds`
* **Total Evaluation Wall Time:** `2,044.28 seconds` (~34.1 minutes)

### 4.2 Per-Landmark Category Summary Table

| Ground Truth | Total_Images | Correct | Accuracy_Pct (%) | Avg_Keypoints | Avg_Matches | Avg_Inliers | Avg_Time_s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `christ_the_reedemer` | 65 | 47 | 72.31% | 344.9 | 84.0 | 60.2 | 4.9642 |
| `eiffel_tower` | 79 | 38 | 48.10% | 410.4 | 86.1 | 41.2 | 5.8513 |
| `great_wall_of_china` | 79 | 75 | 94.94% | 435.5 | 100.5 | 50.0 | 6.2200 |
| `roman_colosseum` | 79 | 44 | 55.70% | 491.3 | 112.3 | 73.5 | 6.9555 |
| `taj_mahal` | 32 | 11 | 34.38% | 409.0 | 104.4 | 48.2 | 5.8349 |
| `unknown` | 5 | 0 | 0.00% | 410.4 | 82.0 | 0.0 | 5.8614 |

---

### 4.3 Representative Experimental Results Table (Sample per Category)

| Ground Truth | Query File | Condition | Keypoints | Reliable Matches | RANSAC Inliers | Predicted Landmark | Localization | Decision | Correct? | Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| `christ_the_reedemer` | `00d2585f75.jpg` | `00d2585f75` | 501 | 90 | 87 | `christ_the_reedemer` | Success | Recognized | Yes | 10.2797 |
| `christ_the_reedemer` | `07f0b2ec5b.jpg` | `07f0b2ec5b` | 225 | 133 | 131 | `christ_the_reedemer` | Success | Recognized | Yes | 3.2657 |
| `christ_the_reedemer` | `0ca3b8d570.jpg` | `0ca3b8d570` | 348 | 123 | 122 | `christ_the_reedemer` | Success | Recognized | Yes | 5.0067 |
| `christ_the_reedemer` | `0d00878384.jpg` | `0d00878384` | 485 | 55 | 54 | `christ_the_reedemer` | Success | Recognized | Yes | 6.8629 |
| `christ_the_reedemer` | `0e0236c5be.jpg` | `0e0236c5be` | 331 | 80 | 73 | `christ_the_reedemer` | Success | Recognized | Yes | 4.7018 |
| `christ_the_reedemer` | `10d04f4650.jpg` | `10d04f4650` | 500 | 79 | 75 | `christ_the_reedemer` | Success | Recognized | Yes | 7.1844 |
| `eiffel_tower` | `0170656bec.jpg` | `0170656bec` | 501 | 88 | 74 | `eiffel_tower` | Success | Recognized | Yes | 7.1493 |
| `eiffel_tower` | `01e2f36bd9.jpg` | `01e2f36bd9` | 57 | 32 | 30 | `eiffel_tower` | Success | Recognized | Yes | 0.9024 |
| `eiffel_tower` | `0e70c2acf7.jpg` | `0e70c2acf7` | 427 | 45 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 5.9520 |
| `eiffel_tower` | `16bca5f59f.jpg` | `16bca5f59f` | 277 | 55 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 3.9510 |
| `eiffel_tower` | `1a2d5256fc.jpg` | `1a2d5256fc` | 244 | 106 | 105 | `eiffel_tower` | Success | Recognized | Yes | 3.3916 |
| `eiffel_tower` | `1b32a19bf1.jpg` | `1b32a19bf1` | 500 | 83 | 61 | `eiffel_tower` | Success | Recognized | Yes | 7.1897 |
| `great_wall_of_china` | `044c6b5c92.jpg` | `044c6b5c92` | 131 | 23 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | Yes | 2.0411 |
| `great_wall_of_china` | `06ee855d59.jpg` | `06ee855d59` | 423 | 85 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | Yes | 6.0708 |
| `great_wall_of_china` | `0ac0152353.jpg` | `0ac0152353` | 500 | 106 | 105 | `great_wall_of_china` | Success | Recognized | Yes | 7.2030 |
| `great_wall_of_china` | `10326c5883.jpg` | `10326c5883` | 440 | 79 | 76 | `great_wall_of_china` | Success | Recognized | Yes | 6.3025 |
| `great_wall_of_china` | `174c8a70ae.jpg` | `174c8a70ae` | 105 | 34 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | Yes | 1.4829 |
| `great_wall_of_china` | `188adbaa13.jpg` | `188adbaa13` | 500 | 122 | 110 | `great_wall_of_china` | Success | Recognized | Yes | 7.1255 |
| `roman_colosseum` | `03ce48b8dd.jpg` | `03ce48b8dd` | 476 | 45 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 6.6565 |
| `roman_colosseum` | `071fadc01d.jpg` | `071fadc01d` | 500 | 103 | 96 | `roman_colosseum` | Success | Recognized | Yes | 7.1723 |
| `roman_colosseum` | `0a1b9a743e.jpg` | `0a1b9a743e` | 500 | 74 | 69 | `roman_colosseum` | Success | Recognized | Yes | 6.9249 |
| `roman_colosseum` | `0cbf8a1fa1.jpg` | `0cbf8a1fa1` | 500 | 102 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 7.0879 |
| `roman_colosseum` | `0d0a715cce.jpg` | `0d0a715cce` | 502 | 101 | 90 | `roman_colosseum` | Success | Recognized | Yes | 6.9470 |
| `roman_colosseum` | `0fb90df2aa.jpg` | `0fb90df2aa` | 500 | 56 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 7.1533 |
| `taj_mahal` | `007b3d98fe.jpg` | `007b3d98fe` | 307 | 124 | 121 | `taj_mahal` | Success | Recognized | Yes | 4.3619 |
| `taj_mahal` | `07d16813c8.jpg` | `07d16813c8` | 396 | 50 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 5.7656 |
| `taj_mahal` | `099307bc2d.jpg` | `099307bc2d` | 225 | 48 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 3.2094 |
| `taj_mahal` | `153d931308.jpg` | `153d931308` | 402 | 80 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 5.8575 |
| `taj_mahal` | `26bebd412f.jpg` | `26bebd412f` | 381 | 94 | 76 | `taj_mahal` | Success | Recognized | Yes | 5.3959 |
| `taj_mahal` | `28ec231502.jpg` | `28ec231502` | 500 | 91 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 7.1279 |
| `unknown` | `0016510b5c.jpg` | Unknown Landmark | 500 | 151 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 7.1250 |
| `unknown` | `01260f7939.jpg` | Unknown Landmark | 207 | 50 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 2.9991 |
| `unknown` | `01a2593196.jpg` | Unknown Landmark | 345 | 48 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 4.9719 |
| `unknown` | `0282f57aab.jpg` | Unknown Landmark | 500 | 109 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 7.1382 |
| `unknown` | `032a2ecea0.jpg` | Unknown Landmark | 500 | 52 | 0 | `great_wall_of_china` | Unreliable Geometric Transformation (Homography Failed) | Recognized | No | 7.0727 |

---

## 5. Discussion of System Performance & Technical Limitations

### 5.1 Strengths of the SIFT + RANSAC Pipeline
1. **High Precision Under True Geometric Correspondence:** Whenever the system successfully estimated a valid homography (`Localization = Success`), the recognition accuracy was **100%**, with extraordinarily high RANSAC inlier ratios (e.g., `131/133` inliers on `07f0b2ec5b.jpg`, `105/106` on `1a2d5256fc.jpg`, and `121/124` on `007b3d98fe.jpg`).
2. **Scale, Rotation, and Illumination Invariance:** Across query images exhibiting severe downscaling (e.g., `01e2f36bd9.jpg` with only 57 keypoints), in-plane rotations, and brightness shifts, SIFT's local gradient orientation normalization and Difference-of-Gaussian (DoG) scale-space extrema detection reliably recovered 30 to 130+ geometrically consistent inliers.
3. **Geometric Rejection of False Positives in Localization:** In **100% of the `unknown` landmark queries** (5/5 images) and **100% of the misclassified known queries**, RANSAC returned **0 inliers** (`Unreliable Geometric Transformation (Homography Failed)`), preventing the system from drawing invalid bounding polygons on false matches.

### 5.2 Key Limitations & Root-Cause Error Analysis
1. **The "Hubness" Phenomenon from Degenerate Reference Images (`great_wall_of_china` False Attractions):**
   * **Symptom:** Every single misclassification in the representative sample table—including all 5 `unknown` queries and difficult viewpoint/illumination queries from `taj_mahal`, `eiffel_tower`, and `roman_colosseum`—was falsely attracted to a single foggy, low-contrast grass/path reference image in the `great_wall_of_china` folder.
   * **Root Cause:** In asymmetric KNN matching (`bf_matcher.knnMatch(query_desc, ref_desc, k=2)`), each of the 500 query descriptors independently searches for its 2 nearest neighbors in the reference image. When a reference image contains very few, nearly uniform descriptors (such as a foggy path with only 2–3 low-contrast keypoints), hundreds of diverse query descriptors can have a smaller Euclidean distance to one of those generic points than to the second generic point, passing Lowe's ratio test in a **many-to-one collapse** (visible as a fan of blue lines converging to two single points in the `great_wall_of_china` reference image).
   * **Why `unknown` Queries Were Marked `Recognized`:** Because the many-to-one collapse inflated `num_good` to 48–151 raw matches (exceeding `MATCH_THRESHOLD = 12`), the Stage-4 raw match decision labeled them `Recognized`, and when `cv2.findHomography` subsequently returned `H = None` (0 inliers), the fallback branch reported `Homography Failed` without overriding `status = "Rejected"`.
2. **Class Imbalance in the Reference Database:**
   * `taj_mahal` had only **126 reference images** compared to **312–315 reference images** for `eiffel_tower`, `great_wall_of_china`, and `roman_colosseum`. Consequently, extreme 3D viewpoint queries of the Taj Mahal were less likely to find a matching canonical angle in the reference set, making them vulnerable to the `great_wall_of_china` hub image (`34.38%` accuracy).
3. **Computational Complexity of Brute-Force Search:**
   * Matching 339 query images against 1,322 individual reference images required **448,158 calls** to `BFMatcher.knnMatch`, averaging `6.02s` per query image.

### 5.3 Concrete Recommendations for Production Deployment
1. **Enforce RANSAC Inlier Gating on Final Recognition Status:** Require `H is not None` and `num_inliers >= MIN_INLIERS` (e.g., $\ge 7$ spatially distinct inliers) as a mandatory condition for `Decision = "Recognized"`. Applying this single rule to the current results table immediately boosts **Unknown Rejection Accuracy from 0% to 100%** (`Avg_Inliers = 0.0` across all 5 unknown images).
2. **Enforce 1-to-1 Mutual / Unique Target Matching:** Deduplicate `good_matches` by `m.trainIdx` (keeping only the minimum-distance query match per reference keypoint) before counting reliable matches. This completely eliminates many-to-one hub collapse on foggy/low-texture reference images.
3. **Filter Low-Keypoint Reference Images:** Exclude or apply a `contrastThreshold = 0.04` filter to reference images with fewer than 50 keypoints during Task 2 indexing.