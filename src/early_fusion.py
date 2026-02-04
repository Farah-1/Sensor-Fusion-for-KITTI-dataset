from util import *
from processing import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



def run_evaluation(img_id, model, save_root):

    img_path = os.path.join(DATA_ROOT, 'img', f'{img_id}.png')
    calib_path = os.path.join(DATA_ROOT, 'calib', f'{img_id}.txt')
    lidar_path = os.path.join(DATA_ROOT, 'velodyne', f'{img_id}.pcd')
    label_path = os.path.join(DATA_ROOT, 'label', f'{img_id}.txt')
    
    img = cv2.imread(img_path)
    P2, R0, V2C = load_calib(calib_path)
    pts_3d_velo = read_pcd(lidar_path)
    gt_boxes, dont_cares = load_kitti_label(label_path)

    raw_objects = get_2d_detections(img_path, model)
    final_preds = filter_duplicate_classes(process_kitti_classes(raw_objects, CLASSES_CFG), 0.8)
    final_preds = filter_predictions_by_dontcare(final_preds, dont_cares, iou_threshold=0.3)
    
    pts_2d, depths = project_lidar_to_cam(pts_3d_velo, P2, R0, V2C)
    
    pred_results = []
    for p in final_preds:
        b = p['bbox']
        in_box = (pts_2d[:, 0] >= b[0]) & (pts_2d[:, 0] <= b[2]) & (pts_2d[:, 1] >= b[1]) & (pts_2d[:, 1] <= b[3])
        if not np.any(in_box): continue
        dist = get_robust_dist(depths[in_box])
        pred_results.append({'class': p['class'], 'z': dist, 'bbox': b})

    tp, fp, correct_class = 0, 0, 0
    matched_gt = set()
    dist_errors, ious = [], []

    for p in pred_results:
        best_iou, best_idx = 0, -1
        for i, g in enumerate(gt_boxes):
            if i in matched_gt: continue
            iou = calculate_iou(p['bbox'], g['bbox'])
            if iou > best_iou: 
                best_iou = iou
                best_idx = i
        
        if best_iou > 0.5:
            tp += 1
            matched_gt.add(best_idx)
            ious.append(best_iou)
            dist_errors.append(abs(p['z'] - gt_boxes[best_idx]['dist']))
            if p['class'].lower() == gt_boxes[best_idx]['type'].lower():
                correct_class += 1
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gt)
    
    metrics = {
        'filename': img_id,
        'precision': tp/(tp+fp+1e-6),
        'recall': tp/(tp+fn+1e-6),
        'mIoU': np.mean(ious) if ious else 0,
        'class_acc': correct_class/(tp+1e-6),
        'mae_dist': np.mean(dist_errors) if dist_errors else 0
    }

    frame_save_dir = os.path.join(save_root, img_id)
    os.makedirs(frame_save_dir, exist_ok=True)
    plot_early_fusion_dashboard(img, raw_objects, final_preds, gt_boxes, dont_cares, pred_results, metrics, img_id, frame_save_dir)

    return metrics

if __name__ == "__main__":
    
    DATA_ROOT = os.path.join(parent_dir, '3DCV01_data')
    MODEL_PATH = os.path.join(parent_dir, 'yolo26n.pt')
    CLASSES_CFG = {'person': 0, 'bicycle': 1, 'car': 2, 'truck': 7}

    ids = ['000031', '000035', '000060', '000080', '000134']
    model = AutoDetectionModel.from_pretrained('ultralytics', MODEL_PATH, confidence_threshold=0.25, device='cpu')
    save_root = os.path.join(parent_dir, 'plots')
    
    all_results = []
    for i in ids:
        print(f"Evaluating Frame {i}...")
        res = run_evaluation(i, model, save_root)
        all_results.append(res)
    
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_root, "early_fusion_detailed_eval.csv"), index=False)
    
    print("\n--- Final Evaluation Summary ---")
    print(df.drop(columns='filename').mean())