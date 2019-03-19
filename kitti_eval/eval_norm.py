import numpy as np
import argparse
from depth_evaluation_utils import compute_error_3d

parser = argparse.ArgumentParser()

parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--gt_file", type=str, help="Path to the Ground truth file")

parser.add_argument('--min_limit', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_limit', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()

def main():
    gt_norm = np.load(args.gt_file)
    pred_norm = np.load(args.pred_file)
    print("Normal prediction and groundtruth loaded...")
    print(gt_norm.shape)
    num_test = gt_norm.shape[0]

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    for i in range(num_test):    
        gt_normi = gt_norm[i]
        
        pred_normi = np.copy(pred_norm[i])
        mask_norm = sum([pred_normi[:,:,i]**2 for i in range(3)])
        
        mask = np.logical_and(mask_norm > args.min_limit, 
                            mask_norm < args.max_limit)
        
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_normi.shape[:2]
        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                        0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
    #     mask = np.ones([gt_height, gt_width]).astype(bool)
        # Scale matching
    #     scalor = np.median(gt_normi[mask])/np.median(pred_normi[mask])
    #     pred_normi[mask] *= scalor

        pred_normi[mask_norm < args.min_limit,:] = [args.min_limit]*3
        pred_normi[mask_norm > args.max_limit,:] = [args.max_limit]*3
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_error_3d(gt_normi[mask], pred_normi[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

main()