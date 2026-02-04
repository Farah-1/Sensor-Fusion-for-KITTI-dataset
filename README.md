# Multi-Modal 3D Object Detection & Distance Estimation
### Autonomous Driving Perception Pipeline (KITTI Dataset)

This project implements and compares two fundamental sensor fusion strategies—**Early Fusion** and **Late Fusion**—using **YOLO (via SAHI)** for 2D vision and **PointPillars** for 3D LiDAR processing. The goal is to accurately detect objects and estimate their 3D distance ($x, y, z$) from the sensor origin.

---

## Project Overview
The pipeline integrates LiDAR point clouds and Camera RGB images to overcome the limitations of single-modality systems, such as camera depth inaccuracies or LiDAR's low semantic resolution.

### Fusion Strategies:
1.  **Early Fusion (Data-Level):**
    * Uses 2D detections (YOLO26n + SAHI) to create frustums in the LiDAR point cloud.
    * Distance is estimated by calculating a robust mean ($3D$ Norm) of the points filtered within the 2D bounding box.
2.  **Late Fusion (Result-Level):**
    * Runs two independent high-performance models: **YOLO** (2D) and **PointPillars** (3D).
    * Matches results using IoU-based association to provide a final detection with both semantic labels and precise 3D localization.

---

##  Key Features
* **SAHI (Sliced Aided Hyper Inference):** Integrated to detect small or distant objects by slicing high-resolution images into $320 \times 320$ segments.
* **Robust Distance Estimation:** Implemented an **IQR (Interquartile Range)** outlier removal filter to eliminate ground plane interference and sensor noise.
* **KITTI Integration:** Full support for KITTI calibration matrices ($P_2, R_0, V2C$) to ensure sub-pixel projection accuracy.
* **Evaluation Dashboard:** Automatically generates a 5-pane visualization for every frame, showing the entire transformation pipeline and real-time metrics.

---

## 🛠 Mathematical Foundation

### 1. 3D to 2D Projection
To project a LiDAR point $P_{L} = [x, y, z, 1]^T$ onto the image plane $p = [u, v, 1]^T$:
$$p = P_2 \times R_{0\_rect} \times Tr_{velo\_to\_cam} \times P_{L}$$
The resulting coordinates are normalized by depth:
$$u = \frac{X_{img}}{Z_{img}}, \quad v = \frac{Y_{img}}{Z_{img}}$$

### 2. Robust Distance (3D Norm)
Instead of relying on simple depth ($Z$), we calculate the Euclidean distance from the sensor:
$$\text{Distance} = \sqrt{x^2 + y^2 + z^2}$$
Outliers are removed using the IQR method:
* $\text{Bounds} = [Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$

---

## Performance Metrics
The system evaluates each frame against KITTI Ground Truth using:
* **Precision & Recall:** Measures detection reliability.
* **mIoU (Mean Intersection over Union):** Measures box overlap accuracy.
* **MAE Dist (Mean Absolute Error):** Average error in meters between predicted and actual distance.
* **Class Accuracy:** Ensures correct classification of Cars, Pedestrians, and Cyclists.

---

## Project Structure
```bash
├── 3DCV01_data
├── src/
│   ├── early_fusion.py      # Pipeline for Data-level fusion
│   ├── late_fusion.py       # Pipeline for Result-level fusion
│   ├── preprocessing.py     # Class merging, NMS, and projection logic
│   └── util.py              # Calibration loading and visualization tools
├── plots/                 # Output plots and metrics logs
├── other pointpillares modules 
└── requirements.txt

```

## Setup & Installation



This project extends the pointpillars  architecture. To run this pipeline:

1. **Clone the late fusion  Base Model:**
   Download/Clone the base PointPillars repository from 
   https://github.com/zhulf0804/PointPillars.
   
3. **Prepare Dataset:**
   Download the KITTI Object Detection dataset and place it in the `/3DCV01_data` directory.

4. **Add My Logic:**
   The scripts in this repository (`early_fusion.py`, `late_fusion.py`, etc.) should be placed in the `/src`  directory .

1- install requirments  
```bash
pip install -r requirements.txt
```
2- Run Evaluation:

For Early Fusion: python src/early_fusion.py

For Late Fusion: python src/late_fusion.py


##  Results & Visualization

Below is the output of the **Late Fusion vs Early Fusion** 

![EARLY FUSION ](plots\000134\earlyFusion_000134.png)
![LATE FUSION ](plots/000134/late_fusion_000134.png)

### Strategy Comparison: Early Fusion vs. Late Fusion

The following table compares the performance metrics of the two fusion strategies.

| Filename | Strategy | Precision | Recall | mIoU | Class Acc | MAE Dist (m) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **000031** | Early Fusion | 1.000 | 1.000 | 0.831 | 0.833 | 1.255 |
| | **Late Fusion** | **1.000** | **0.667** | **0.850** | **1.000** | **0.829** |
| --- | --- | --- | --- | --- | --- | --- |
| **000035** | Early Fusion | 0.800 | 0.800 | 0.857 | 1.000 | 1.533 |
| | **Late Fusion** | **0.800** | **0.800** | **0.857** | **1.000** | **3.659** |
| --- | --- | --- | --- | --- | --- | --- |
| **000060** | Early Fusion | 0.600 | 0.750 | 0.853 | 0.667 | 3.303 |
| | **Late Fusion** | **1.000** | **0.250** | **0.900** | **1.000** | **0.295** |
| --- | --- | --- | --- | --- | --- | --- |
| **000080** | Early Fusion | 1.000 | 1.000 | 0.830 | 1.000 | 0.980 |
| | **Late Fusion** | **1.000** | **0.750** | **0.883** | **1.000** | **0.405** |
| --- | --- | --- | --- | --- | --- | --- |
| **000134** | Early Fusion | 0.833 | 0.667 | 0.752 | 0.900 | 4.075 |
| | **Late Fusion** | **0.750** | **0.400** | **0.812** | **1.000** | **1.922** |

### Technical Analysis

1. **Precision vs. Recall Trade-off**:
   - **Early Fusion** shows higher **Recall** (avg ~0.84). Because it starts with 2D vision, it captures more objects, even if the LiDAR data is sparse.
   - **Late Fusion** shows higher **Precision** (avg ~0.91) but lower Recall. This is because it requires *both* models to agree on an object before confirming it.

2. **Distance Accuracy**:
   - **Late Fusion** significantly outperforms in **MAE Dist** (e.g., Frame 000060 reduced error from 3.3m to 0.29m). Using a dedicated 3D model (PointPillars) provides much better spatial localization than 2D-to-3D back-projection.