from models.nfo_model import nfoModel
import nibabel as nib
from options.test_options import TestOptions
import torch
import os
import csv
import time
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    gpu_ids = opt.gpu_ids
    weights_path = opt.weights_path

    use_cuda = len(str(gpu_ids).strip()) > 0 and torch.cuda.is_available()

    records = []  # (subject, volume_shape, time_s, img_per_s)

    for item in os.listdir(opt.fod_path):
        fod_path = os.path.join(opt.fod_path, item, "input/ssmtCSD_norm_WM.nii.gz")
        brain_mask_path = os.path.join(opt.fod_path, item, "mask/nodif_brain_mask.nii.gz")
        output_path = os.path.join(opt.output_path, item, f"predicted_fod.nii.gz")
        print(item)

        fodlr_file = nib.load(fod_path)
        brain_mask_file = nib.load(brain_mask_path)

        fixed_fodlr_affine = fodlr_file.affine
        fixed_fodlr = fodlr_file.get_fdata()
        fixed_brain_mask = brain_mask_file.get_fdata()

        fixed_fodlr = fixed_fodlr.copy() ##
        fixed_brain_mask = fixed_brain_mask.copy() ##

        assert fixed_fodlr.shape[:3] == fixed_brain_mask.shape, \
            'Input fod and mask should have the same shape'

        print(f"fixed_fodlr:{fixed_fodlr.shape}")

        model = nfoModel(opt)
        model.load_weights(weights_path=weights_path)
        model.eval()

        model.set_input_for_test(fixed_fodlr, fixed_brain_mask,
                                 fixed_fodlr_affine, fodlr_file.header)

        if use_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        model.our_fast_test(output_path)

        if use_cuda:
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        speed = 1.0 / elapsed
        records.append((item, fixed_fodlr.shape[:3], elapsed, speed))
        print(f'[{item}] total:{elapsed:.3f} s/sample, v: {speed:.4f} sample/s')


