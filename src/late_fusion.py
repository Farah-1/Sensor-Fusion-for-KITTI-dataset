from util import *
from processing import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pointpillars.model import PointPillars

def run_full_evaluation(img_id,model_2d,model_3d):

    img_path = os.path.join(DATA_ROOT, 'img', f'{img_id}.png')
    calib_path = os.path.join(DATA_ROOT, 'calib', f'{img_id}.txt')
    lidar_path = os.path.join(DATA_ROOT, 'velodyne', f'{img_id}.pcd')
    label_path = os.path.join(DATA_ROOT, 'label', f'{img_id}.txt')
    
   
    img = cv2.imread(img_path)
    P2, R0, V2C = load_calib(calib_path)
    pc = read_pcd(lidar_path)
    gt_objs, dont_cares = load_kitti_label(label_path)

    raw_2d = get_2d_detections(img_path, model_2d)
    final_2d = filter_duplicate_classes(process_kitti_classes(raw_2d, {'person': 0, 'bicycle': 1, 'car': 2, 'truck': 7}), 0.8)
    final_2d = filter_predictions_by_dontcare(final_2d, dont_cares, iou_threshold=0.3)
    with torch.no_grad():
        res_3d = model_3d(batched_pts=[torch.from_numpy(pc).to(device)], mode='test')[0]
    
    fused_res = []
    for p2d in final_2d:
        for box3d in res_3d['lidar_bboxes']:
            proj = get_2d_bbox_from_3d(box3d, P2, R0, V2C, img.shape)
            if proj and calculate_iou(p2d['bbox'], proj) > 0.3:
                fused_res.append({'class': p2d['class'], 'bbox': p2d['bbox'], 'dist': np.linalg.norm(box3d[:3])})
                break
  
    metrics = calculate_metrics(img_id,fused_res, gt_objs, dont_cares)


    frame_save_dir = os.path.join(save_root, img_id)
    os.makedirs(frame_save_dir, exist_ok=True)
    plot_late_fusion_results(img, raw_2d, final_2d, res_3d['lidar_bboxes'], fused_res, gt_objs, dont_cares,metrics, img_id, frame_save_dir, P2, R0, V2C)
    return metrics
   
if __name__ == '__main__':
  
    DATA_ROOT = os.path.join(parent_dir, '3DCV01_data')
    MODEL_PATH = os.path.join(parent_dir, 'yolo26n.pt')
    CLASSES_CFG = {'person': 0, 'bicycle': 1, 'car': 2, 'truck': 7}

    ids = ['000031', '000035', '000060', '000080', '000134']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pp_ckpt = "pretrained/epoch_160.pth"
    model_2d = AutoDetectionModel.from_pretrained('ultralytics', MODEL_PATH, confidence_threshold=0.25, device='cpu')
    model_3d = PointPillars(nclasses=3).to(device)
    model_3d.load_state_dict(torch.load(pp_ckpt, map_location=device))
    model_3d.eval()
   
    save_root = os.path.join(parent_dir, 'plots')
    
    all_results = []
    for i in ids:
        print(f"Evaluating Frame {i}...")
        res = run_full_evaluation(i, model_2d, model_3d)
        all_results.append(res)
    
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(save_root, "late_fusion_detailed_eval.csv"), index=False)
    
    print("\n--- Final Evaluation Summary ---")
    print(df.drop(columns='filename').mean())

