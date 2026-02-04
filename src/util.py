import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN
import sys
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from sahi import AutoDetectionModel

def load_calib(calib_path):
    """
    Parses the KITTI calibration file to extract projection and transformation matrices.
    
    Steps:
    1. Read 'P2' (Camera 2 projection matrix) and reshape to (3, 4).
    2. Read 'R0_rect' (Rectification matrix) and reshape to (3, 3).
    3. Read 'Tr_velo_to_cam' (LiDAR to Camera transformation) and reshape to (3, 4).
    4. Convert R0 and V2C to 4x4 homogeneous matrices to allow matrix multiplication 
     
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' not in line: continue
            key, val = line.split(':')
            calib[key] = np.array([float(x) for x in val.split()])
    
    P2 = calib['P2'].reshape(3, 4)
    R0 = calib['R0_rect'].reshape(3, 3)
    V2C = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = R0
    
    Velo2Cam = np.eye(4)
    Velo2Cam[:3, :4] = V2C
    
    return P2, R0_rect, Velo2Cam

def load_kitti_label(label_path):
    """
    Reads KITTI ground truth labels from a text file.
    
    Returns:
    - objects: List of valid detectable objects (Car, Pedestrian, etc.).
    - dont_cares: List of regions marked as 'DontCare' to be ignored during evaluation.
    """
    objects, dont_cares = [], []
    if not os.path.exists(label_path): return objects, dont_cares
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            obj = {
                'type': data[0],
                'bbox': [float(data[4]), float(data[5]), float(data[6]), float(data[7])],
                'dist': float(data[13]) if len(data) > 13 else -1
            }
            if obj['type'].lower() == 'dontcare': dont_cares.append(obj)
            else: objects.append(obj)
    return objects, dont_cares

def parse_gt(label_path):
    """
    Simplified parser for KITTI ground truth focusing on specific object classes.
    """
    bboxes = []
    kitti_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
    if not os.path.exists(label_path): return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            p = line.split()
            if p[0] in kitti_classes:
                bboxes.append({'class': p[0], 'bbox': [float(x) for x in p[4:8]], 'z': float(p[13])})
    return bboxes

def read_pcd(pcd_path):
    """
    Reads a .pcd point cloud file using Open3D and ensures it has 4 channels (x, y, z, intensity).
    Adds a zero-filled intensity channel if only 3 channels are present.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points).astype(np.float32)
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, intensity], axis=1)
    return points

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two 2D bounding boxes.
    
     Steps:
    1. Determine the coordinates of the intersection rectangle.
    2. Calculate Intersection Area
    4. calculate IoU 
    """
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(area1 + area2 - intersection_area)

def get_union_box(box1, box2):
    """
    Computes the smallest bounding box that contains both box1 and box2.
    """
    return [min(box1[0], box2[0]), min(box1[1], box2[1]), 
            max(box1[2], box2[2]), max(box1[3], box2[3])]

def project_3d_to_2d(pts_3d, P2, R0, V2C):
    """
    Projects 3D LiDAR points into 2D image coordinates.
    
    Mathematical Steps:
    1. Convert 3D points (N, 3) to Homogeneous coordinates (N, 4).
    2. Transform points from LiDAR frame to Camera frame
    3. Filter points where Z < 0.1 (points behind or too close to the camera).
    4. Project to Image Plane using Intrinsic Matrix
    5. Normalize (u, v) coordinates by dividing by the depth Z.
    """
    pts_3d_hom = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
    pts_3d_cam = pts_3d_hom @ V2C.T @ R0.T
    mask = pts_3d_cam[:, 2] > 0.1
    pts_2d = pts_3d_cam[mask] @ P2.T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    return pts_2d[:, :2], mask

def get_2d_bbox_from_3d(lidar_box, P2, R0, V2C, img_shape):
    """
    Calculates a 2D bounding box by projecting the 8 corners of a 3D LiDAR box into the image.
    
    Mathematical Steps:
    1. Define 8 corners of a 3D box based on length, width, and height.
    2. Rotate corners by Yaw angle and translate to the box center (x, y, z).
    3. Project corners to 2D using project_3d_to_2d.
    4. Find the min/max (u, v) to form the 2D bounding box [x1, y1, x2, y2].
    """
    x, y, z, l, w, h, yaw = lidar_box
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h, h, h, h, 0, 0, 0, 0]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    corners_3d = (rot_mat @ corners_3d) + np.array([[x], [y], [z]])
    pts_2d, _ = project_3d_to_2d(corners_3d.T, P2, R0, V2C)
    if len(pts_2d) < 4: return None
    return [max(0, np.min(pts_2d[:, 0])), max(0, np.min(pts_2d[:, 1])), 
            min(img_shape[1], np.max(pts_2d[:, 0])), min(img_shape[0], np.max(pts_2d[:, 1]))]

def plot_late_fusion_results(img, raw_2d, final_2d, lidar_boxes, fused_results, gt_objs, dont_cares, metrics, filename, save_path, P2, R0, V2C):
    """
    Visualizes the complete Late Fusion pipeline across 5 subplots and saves the dashboard.
    Now includes a statistical metrics legend for performance evaluation.
    """
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 2)
    
    # --- 1. Raw 2D Detections ---
    ax0 = fig.add_subplot(gs[0, 1])
    img_raw = img.copy()
    for r in raw_2d:
        b = [r.bbox.minx, r.bbox.miny, r.bbox.maxx, r.bbox.maxy]
        cv2.rectangle(img_raw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
    ax0.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"1. Raw 2D Detections ({len(raw_2d)} objects)")

    # --- 2. Corrected 2D ---
    ax1 = fig.add_subplot(gs[1,:])
    img_corr = img.copy()
    for f in final_2d:
        b = f['bbox']
        cv2.rectangle(img_corr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
        cv2.putText(img_corr, f['class'], (int(b[0]), int(b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    ax1.imshow(cv2.cvtColor(img_corr, cv2.COLOR_BGR2RGB))
    ax1.set_title("2. Corrected 2D (Merged & Duplicates Removed)")

    # --- 3. 3D LiDAR Projections ---
    ax2 = fig.add_subplot(gs[2,:])
    img_3d = img.copy()
    for box in lidar_boxes:
        proj = get_2d_bbox_from_3d(box, P2, R0, V2C, img.shape)
        if proj:
            dist = np.linalg.norm(box[:3])
            cv2.rectangle(img_3d, (int(proj[0]), int(proj[1])), (int(proj[2]), int(proj[3])), (255, 128, 0), 2)
            cv2.putText(img_3d, f"{dist:.1f}m", (int(proj[0]), int(proj[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    ax2.imshow(cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB))
    ax2.set_title("3. 3D LiDAR Projections (Meters on top)")

    # --- 4. Ground Truth ---
    ax3 = fig.add_subplot(gs[0, 0])
    img_gt = img.copy()
    for g in gt_objs:
        b = g['bbox']
        cv2.rectangle(img_gt, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(img_gt, f"{g['type']} {g['dist']:.1f}m", (int(b[0]), int(b[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for dc in dont_cares:
        b = dc['bbox']
        cv2.rectangle(img_gt, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    ax3.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    ax3.set_title("4. Ground Truth (Objects=Green, DontCare=Red)")

    # --- 5. Final Late Fusion Result ---
    ax4 = fig.add_subplot(gs[3, :])
    img_fused = img.copy()
    for f in fused_results:
        b = f['bbox']
        cv2.rectangle(img_fused, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
        label = f"{f['class']} {f['dist']:.1f}m"
        cv2.putText(img_fused, label, (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    ax4.imshow(cv2.cvtColor(img_fused, cv2.COLOR_BGR2RGB))
    ax4.set_title(f"5. Final Late Fusion Result - Frame: {filename}")

    # --- Metrics Legend ---
    info_text = (
        f"LATE FUSION METRICS FOR {filename}:\n"
        f"---------------------------------\n"
        f"Precision: {metrics['precision']:.2f}\n"
        f"Recall:    {metrics['recall']:.2f}\n"
        f"Avg IoU:   {metrics['mIoU']:.2f}\n"
        f"Class Acc: {metrics['class_acc']:.2%}\n"
        f"MAE Dist:  {metrics['mae_dist']:.2f}m"
    )
    fig.text(0.85, 0.5, info_text, fontsize=14, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1'))

    plt.tight_layout()
    dashboard_path = os.path.join(save_path, f"late_fusion_{filename}.png")
    plt.savefig(dashboard_path, dpi=120, bbox_inches='tight')
    plt.close()

    
def get_robust_dist(depths_in_box):
    """
    Estimates a robust distance by removing statistical outliers using the Interquartile Range (IQR).
    
    Mathematical Steps:
    1. Calculate Q1 (25th percentile) and Q3 (75th percentile).
    2. Compute IQR = Q3 - Q1$.
    3. Define bounds: [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
    4. Filter values outside these bounds and return the mean of the remaining data.
    """
    if len(depths_in_box) == 0: return 0
    if len(depths_in_box) == 1: return depths_in_box[0]
    
    q1, q3 = np.percentile(depths_in_box, 25), np.percentile(depths_in_box, 75)
    iqr = q3 - q1
    mask_clean = (depths_in_box >= (q1 - 1.5 * iqr)) & (depths_in_box <= (q3 + 1.5 * iqr))
    filtered = depths_in_box[mask_clean]
    return np.mean(filtered) if len(filtered) > 0 else np.mean(depths_in_box)

def plot_early_fusion_dashboard(img_orig, raw_2d, final_2d, gt_boxes, dont_cares, pred_results, metrics, filename, save_dir):
    """
    Creates a comprehensive dashboard for the Early Fusion pipeline.
    Displays Ground Truth, Raw YOLO output, Filtered 2D Boxes, and the Fusion result with a metrics legend.
    """
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 2)

    ax3 = fig.add_subplot(gs[0, 0])
    img_gt = img_orig.copy()
    for g in gt_boxes:
        b = g.get('bbox') or g.get('bbox_2d')
        cv2.rectangle(img_gt, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(img_gt, f"{g.get('type','Obj')} {g.get('dist',0):.1f}m", (int(b[0]), int(b[1]-5)), 0, 0.6, (0, 255, 0), 2)
    for dc in dont_cares:
        b = dc.get('bbox') or dc.get('bbox_2d')
        cv2.rectangle(img_gt, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    ax3.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)); ax3.set_title("1. Ground Truth (DontCare=Red)")

    ax0 = fig.add_subplot(gs[0, 1])
    img_raw = img_orig.copy()
    for r in raw_2d:
        b = [r.bbox.minx, r.bbox.miny, r.bbox.maxx, r.bbox.maxy]
        cv2.rectangle(img_raw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
    ax0.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)); ax0.set_title("2. Raw YOLO Detections")

    ax1 = fig.add_subplot(gs[1, :])
    img_corr = img_orig.copy()
    for p in final_2d:
        b = p['bbox']
        cv2.rectangle(img_corr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
    ax1.imshow(cv2.cvtColor(img_corr, cv2.COLOR_BGR2RGB)); ax1.set_title("3. Filtered & Corrected 2D Boxes")

    ax4 = fig.add_subplot(gs[2, :])
    img_fused = img_orig.copy()
    for p in pred_results:
        b = p['bbox']
        cv2.rectangle(img_fused, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
        cv2.putText(img_fused, f"{p['class']} {p['z']:.1f}m", (int(b[0]), int(b[1]-10)), 0, 0.8, (0, 255, 0), 2)
    ax4.imshow(cv2.cvtColor(img_fused, cv2.COLOR_BGR2RGB)); ax4.set_title(f"4. Final Fusion Result - Frame: {filename}")

    info_text = (
        f"METRICS FOR {filename}:\n"
        f"----------------------\n"
        f"Precision: {metrics['precision']:.2f}\n"
        f"Recall:    {metrics['recall']:.2f}\n"
        f"Avg IoU:   {metrics['mIoU']:.2f}\n"
        f"Class Acc: {metrics['class_acc']:.2%}\n"
        f"MAE Dist:  {metrics['mae_dist']:.2f}m"
    )
    fig.text(0.85, 0.5, info_text, fontsize=14, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"earlyFusion_{filename}.png"), bbox_inches='tight')
    plt.close()