from util import *
from sahi.predict import get_sliced_prediction

def filter_duplicate_classes(preds, iou_threshold=0.8):
    """
    Performs Class-Agnostic Non-Maximum Suppression (NMS) to remove overlapping detections.
    
    Mathematical Steps:
    1. Sort all predictions by confidence score in descending order.
    2. Iterate through predictions and compare the current box with already "kept" boxes.
    3. Calculate IoU between the current box and each kept box.
    4. If IoU > Threshold$, the box with the lower score is discarded as a duplicate.
    """
    if not preds:
        return []

    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    keep = []
    
    for i, p1 in enumerate(preds):
        discard = False
        for j, p2 in enumerate(keep):
            iou = calculate_iou(p1['bbox'], p2['bbox'])
            if iou > iou_threshold:
                discard = True
                break
        
        if not discard:
            keep.append(p1)
            
    return keep

def filter_predictions_by_dontcare(pred_results, dont_cares, iou_threshold=0.4):
    """
    Filters out detections that fall into 'DontCare' regions defined in KITTI labels.
    
    Logic:
    - For each prediction, calculate IoU with all 'DontCare' bounding boxes.
    - If the overlap exceeds the threshold, the prediction is considered to be in an 
      unlabeled or ambiguous region and is removed from evaluation.
    """
    clean_predictions = []
    
    for p in pred_results:
        is_ignored = False
        p_bbox = p.get('bbox') or p.get('bbox_2d')
        
        for dc in dont_cares:
            dc_bbox = dc.get('bbox') or dc.get('bbox_2d')
            
            if p_bbox is not None and dc_bbox is not None:
                iou = calculate_iou(p_bbox, dc_bbox)
                if iou > iou_threshold:
                    is_ignored = True
                    break 
        
        if not is_ignored:
            clean_predictions.append(p)
            
    return clean_predictions

def project_lidar_to_cam(pts_3d_velo, P2, R0, V2C):
    """
    Projects 3D LiDAR point cloud into 2D camera coordinates using KITTI matrices.
    
    Mathematical Steps:
    1. Prepare homogeneous 3D coordinates: P_{velo} = [X, Y, Z, 1]^T.
    2. Reshape matrices to proper dimensions for multiplication.
    3. Transformation Pipeline
    4. Filter Depth: Retain points where Z_{cam} > 0.1.
    5. Image Projection
    6. Normalization:
    """
    pts_3d = pts_3d_velo[:, :3] 
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    
    if V2C.size == 12:
        V2C_4x4 = np.eye(4)
        V2C_4x4[:3, :4] = V2C.reshape(3, 4)
    else:
        V2C_4x4 = V2C.reshape(4, 4)

    if R0.size == 9:
        R0_4x4 = np.eye(4)
        R0_4x4[:3, :3] = R0.reshape(3, 3)
    elif R0.size == 16:
        R0_4x4 = R0.reshape(4, 4)
    else:
        R0_4x4 = np.eye(4)
        R0_4x4[:3, :4] = R0.reshape(3, 4)

    pts_3d_cam = pts_3d_hom @ V2C_4x4.T @ R0_4x4.T

    mask = pts_3d_cam[:, 2] > 0.1
    pts_3d_cam_filtered = pts_3d_cam[mask]
    
    P2_3x4 = P2.reshape(-1, 4)[:3, :4]
    pts_2d_hom = pts_3d_cam_filtered @ P2_3x4.T
    
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
    
    return pts_2d, pts_3d_cam_filtered[:, 2]

def get_2d_detections(img_path, detection_model):
    """
    Executes Sliced Analysis of Hierarchical Images (SAHI) for small object detection.
    Segments the image into 320 * 320 slices to improve YOLO's performance on distant objects.
    """
    result = get_sliced_prediction(
        img_path, 
        detection_model, 
        slice_height=320, 
        slice_width=320
    )
    return result.object_prediction_list

def calculate_metrics(img_id, fused_preds, gt_objects, dont_cares):
    """
    Calculates key performance indicators: Precision, Recall, mIoU, Class Accuracy, and MAE.
    
    Mathematical Steps:
    1. Exclude predictions overlapping with 'DontCare' regions.
    2. Match Predictions to GT using IoU. A match is successful if $IoU > 0.5$.
    3. $Precision = TP\{TP + FP}
    4. $Recall ={TP}\{TP + FN}
    5. $MAE_{dist} = abs{ Pred_{dist} - GT_{dist}}\n
    """
    clean_preds = [p for p in fused_preds if not any(calculate_iou(p['bbox'], dc['bbox']) > 0.1 for dc in dont_cares)]
    
    tp, fp, correct_class = 0, 0, 0
    total_dist_err, dist_count = 0, 0
    ious, matched_gt_idx = [], set()

    for p in clean_preds:
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gt_objects):
            if i in matched_gt_idx: continue
            iou = calculate_iou(p['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        
        if best_iou > 0.5: 
            tp += 1
            matched_gt_idx.add(best_idx)
            ious.append(best_iou)
            if p['class'].lower() == gt_objects[best_idx]['type'].lower():
                correct_class += 1
            if gt_objects[best_idx]['dist'] > 0:
                total_dist_err += abs(p['dist'] - gt_objects[best_idx]['dist'])
                dist_count += 1
        else:
            fp += 1
            
    fn = len(gt_objects) - len(matched_gt_idx)
    
    metrics = {
        'filename': img_id,
        'precision': tp / (tp + fp + 1e-6),
        'recall': tp / (tp + fn + 1e-6),
        'mIoU': np.mean(ious) if ious else 0,
        'class_acc': correct_class / (tp + 1e-6),
        'mae_dist': total_dist_err / (dist_count + 1e-6)
    }
    return metrics

def process_kitti_classes(object_list, classes_cfg, iou_threshold=0.8):
    """
    Standardizes object classes for KITTI evaluation.
    
    Special Logic:
    - Cyclist Merging: If a 'Person' and a 'Bicycle' detection overlap, they are merged 
      into a single 'Cyclist' class using a union bounding box.
    - Class Filtering: Applies class-agnostic NMS to resolve multi-class overlaps.
    """
    persons = [p for p in object_list if p.category.id == classes_cfg['person']]
    bikes = [p for p in object_list if p.category.id == classes_cfg['bicycle']]
    vehicles = [p for p in object_list if p.category.id in [classes_cfg['car'], classes_cfg['truck']]]

    final_preds = []
    used_p_indices = set()

    for bike in bikes:
        best_iou, best_p_idx = 0.05, -1
        b_box = bike.bbox.to_voc_bbox()
        for p_idx, person in enumerate(persons):
            if p_idx in used_p_indices: continue
            iou = calculate_iou(person.bbox.to_voc_bbox(), b_box)
            if iou > best_iou:
                best_iou, best_p_idx = iou, p_idx
        
        if best_p_idx != -1:
            merged = get_union_box(persons[best_p_idx].bbox.to_voc_bbox(), b_box)
            score = (bike.score.value + persons[best_p_idx].score.value) / 2
            final_preds.append({'class': 'Cyclist', 'bbox': merged, 'score': score})
            used_p_indices.add(best_p_idx)

    for p_idx, person in enumerate(persons):
        if p_idx not in used_p_indices:
            final_preds.append({
                'class': 'Pedestrian', 
                'bbox': person.bbox.to_voc_bbox(), 
                'score': person.score.value
            })

    for v in vehicles:
        name = "Car" if v.category.id == classes_cfg['car'] else "Truck"
        final_preds.append({
            'class': name, 
            'bbox': v.bbox.to_voc_bbox(), 
            'score': v.score.value
        })

    return filter_duplicate_classes(final_preds, iou_threshold)