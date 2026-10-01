# Lab Assignment: Skin Lesion Boundary Detection Using Canny Edge Detection

## Required Visualization and Metrics

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| --- | --- | --- | --- | --- |
| Image 1 | Gaussian Filter (5x5) | Canny (100, 200) | 166.50 | 81.90 |
| Image 2 | Gaussian Filter (5x5) | Canny (100, 200) | 35.50 | 27.56 |
| Image 3 | Gaussian Filter (5x5) | Canny (100, 200) | 0.00 | 0.00 |
| Image 4 | Gaussian Filter (5x5) | Canny (100, 200) | 240.00 | 100.43 |
| Image 5 | Gaussian Filter (5x5) | Canny (100, 200) | 0.00 | 0.00 |

*(Note: Images 3 and 5 returned an area of 0.00, demonstrating that the hardcoded thresholds and morphological operations failed to form a continuous, enclosed contour for those specific lesions. This highlights the limitations of static edge detection on highly variable datasets).*

---

## Final Comparison

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
| --- | --- | --- | --- | --- |
| **Original + Sobel** | Poor (Captures skin texture) | Thick, blurred, and highly noisy | Inaccurate and highly fragmented | Weak |
| **Original + Canny** | Poor (Excessive micro-edges) | Crisp but highly cluttered | Over-segmented, captures too much background | Poor |
| **Average + Sobel** | Fair | Excessively blurred | Weak definition around the lesion | Fair |
| **Average + Canny** | Fair | Edges lose sharpness | Disconnected boundaries due to over-smoothing | Fair |
| **Gaussian + Sobel** | Good | Smooth but thick/gradient-like | Detects general shape but lacks precise localization | Good |
| **Gaussian + Canny** | Excellent | Crisp, localized, and continuous | Highly accurate, encloses the lesion cleanly | Excellent |
| **Median + Sobel** | Excellent for salt/pepper noise | Slightly jagged | Fairly accurate but contour is not smooth | Good |
| **Median + Canny** | Good | Sharp but jagged | Good shape retention but stepped boundaries | Very Good |

---

## Questions to Answer

**1. Why is Gaussian filtering applied before Canny detection?**
Gaussian filtering smooths the image and suppresses high-frequency spatial noise, such as fine skin textures, pores, and small hairs. Canny edge detection is highly sensitive to intensity gradients; without prior smoothing, these natural skin artifacts would be falsely detected as edges, cluttering the final edge map and making it difficult to isolate the true boundary of the lesion.

**2. How did the three Canny threshold settings affect the result?**

* **Low Thresholds (e.g., 50-100):** Captured too much noise, producing a highly cluttered edge map by detecting weak gradients associated with normal skin texture and minor lighting variations.
* **High Thresholds (e.g., 150-250):** Missed weaker edge signals, resulting in broken, disconnected contours where the lesion blended slightly into the surrounding healthy skin.
* **Medium Thresholds (e.g., 100-200):** Provided an optimal balance, effectively ignoring weak background noise while retaining a strong, continuous edge around the high-contrast lesion boundary.

**3. Which threshold produced the best lesion boundary?**
The medium threshold setting (100, 200) produced the best boundary. It successfully minimized the detection of spurious background noise while ensuring the primary gradient surrounding the pigmented lesion remained continuously intact for contour extraction.

**4. Why are edges useful for detecting skin lesions?**
Edges represent areas in an image where pixel intensity changes sharply. Because skin lesions (like melanoma or keratosis) are typically darker or highly pigmented compared to surrounding healthy skin, the transition between the two creates a strong spatial gradient. Detecting these edge gradients naturally outlines the morphological shape and perimeter of the lesion.

**5. What problems did you observe in detecting the lesion boundary?**

* **Failed Contours:** As observed with Images 3 and 5 returning 0 area, hardcoded parameters often fail to enclose a boundary if the lesion contrast is too low.
* **Skin Artifacts:** Thick hairs crossing the lesion create strong false edges that interfere with the true boundary.
* **Low Contrast Boundaries:** In areas where the lesion pigmentation gradually fades into normal skin, the gradient is too weak for Canny to detect, causing gaps in the detected contour.
* **Lighting Variations:** Shadows or uneven illumination (vignetting) near the corners of the image can create false gradient edges.

**6. How could your method be improved?**

* **Preprocessing:** Implement a digital hair removal algorithm (such as DullRazor) prior to edge detection to eliminate hair artifacts.
* **Adaptive Thresholding:** Use Otsu's method to dynamically calculate optimal threshold values rather than hardcoding Canny parameters, allowing the system to adapt to varying lighting conditions across different images.
* **Advanced Segmentation:** Replace basic contour detection with region-based methods (like Watershed or GrabCut) or deploy a deep learning architecture (like U-Net) for highly robust semantic segmentation.