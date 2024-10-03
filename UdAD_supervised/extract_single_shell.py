import argparse
import os
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from utils.mrtrix import *


def downsampling(input_path, output_path, bval_file, bvec_file, bval=1000, bval_range=20):
    mif = load_mrtrix(input_path)
    print(type(mif))
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)

    print(bvals, bvecs)
    
    lr_bvecs, lr_bvals, lr_index, b0_index = extract_single_shell(
        bvals, bvecs, bval, bval_range, sample=False
    )
    lr_index = np.array(lr_index.tolist())
    lr_bvals = np.array([0] + lr_bvals.tolist())
    lr_bvecs = np.concatenate([np.zeros((1, 3)), lr_bvecs])
    new_grad = np.concatenate([lr_bvecs, lr_bvals.reshape(-1, 1)], axis=-1)

    mif_b0 = np.mean(mif.data[..., b0_index], axis=-1, keepdims=True)
    f_data = np.concatenate([mif_b0, mif.data[..., lr_index]], axis=-1)

    mif.data = f_data
    mif.grad = new_grad

    mif.save(output_path)


def extract_single_shell(
        bvals, bvecs, extract_bval=1000, extract_range=20, directions=32,
        sample=True
):
    # Load original dwi and fslgrad.
    if isinstance(bvals, str) and isinstance(bvecs, str):
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)

    # Obtain the index for the selected bvalue.
    idx = np.logical_and(
        (extract_bval - extract_range) <= gtab.bvals,
        gtab.bvals <= (extract_bval + extract_range)
    )
    idx_0 = np.logical_and(
        (0 - extract_range) <= gtab.bvals,
        gtab.bvals <= (0 + extract_range)
    )

    # Get the extracted bvals and bvecs
    new_bvals = gtab.bvals[idx]
    new_bvecs = gtab.bvecs[idx]

    normalized_bvecs = new_bvecs / np.linalg.norm(new_bvecs, axis=1, keepdims=True)
    bvecs_input = normalized_bvecs.copy()
    bvecs_input[bvecs_input[:, -1] < 0, :] = - bvecs_input[bvecs_input[:, -1] < 0, :]

    # Use function to compute index
    if sample:
        lr_index = kenStone(bvecs_input, directions)

        # Extract low gradient direction resolution data
        lr_bvecs = new_bvecs[lr_index]
        lr_bvals = new_bvals[lr_index]

        lr_index = np.where(idx)[0][lr_index]
    else:
        lr_bvecs = new_bvecs
        lr_bvals = new_bvals

        lr_index = np.where(idx)[0]

    return lr_bvecs, lr_bvals, lr_index, np.where(idx_0)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--bval_file')
    parser.add_argument('--bvec_file')
    parser.add_argument('--bval') 
    args = parser.parse_args()
    downsampling(input_path=args.input_path, output_path=args.output_path, bval_file=args.bval_file, bvec_file=args.bvec_file, bval=int(args.bval))

if __name__ == '__main__':
    main()
