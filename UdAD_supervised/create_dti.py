import os
import subprocess


dwi_folder = '/home/doraemon/Documents/UdAD_supervised/dwi_limited_field_of_view_full'
mask_folder = '/home/doraemon/Documents/UdAD_supervised/converted_masks'
output_root_folder = '/home/doraemon/Documents/UdAD_supervised/dti_artifact'
bval_folder = '/home/doraemon/Documents/UdAD_supervised/bval_artifact'
bvec_folder = '/home/doraemon/Documents/UdAD_supervised/bvec_artifact'

for dwi_file in os.listdir(dwi_folder):
    if dwi_file.endswith('_DWI_processed_b1000.mif.gz'):
        case_id = dwi_file.split('_')[0]

        output_folder = os.path.join(output_root_folder, case_id)
        os.makedirs(output_folder, exist_ok=True)

        dwi_path = os.path.join(dwi_folder, dwi_file)
        mask_file = f'{case_id}_DWI_brainmask.nii.gz'
        mask_path = os.path.join(mask_folder, mask_file)

        single_shell_mif = os.path.join(output_folder, 'single_shell.mif.gz')
        single_shell_nii = os.path.join(output_folder, 'single_shell.nii.gz')
        bvec_file = os.path.join(output_folder, 'single_shell_bvec')
        bval_file = os.path.join(output_folder, 'single_shell_bval')
        dti_prefix = os.path.join(output_folder, 'DTI_eval')

        bval_path = os.path.join(bval_folder, f'{case_id}_bval')
        bvec_path = os.path.join(bvec_folder, f'{case_id}_bvec')


        subprocess.run(['/usr/bin/python3.10', '/home/doraemon/Documents/UdAD_supervised/extract_single_shell.py', '--input_path', dwi_path, '--output_path', single_shell_mif, '--bval_file', bval_path, '--bvec_file', bvec_path, '--bval', '1000'])
        subprocess.run(['/home/doraemon/Downloads/Software/mrtrix3/bin/mrconvert', single_shell_mif, single_shell_nii, '-export_grad_fsl', bvec_file, bval_file])
        subprocess.run(['/home/doraemon/Downloads/Software/FSL/bin/dtifit', '-k', single_shell_nii, '-o', dti_prefix, '-m', mask_path, '-r', bvec_file, '-b', bval_file, '--save_tensor'])

        keep_files = [f'{dti_prefix}_tensor.nii.gz', f'{dti_prefix}_FA.nii.gz']

        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if file_path not in keep_files:
                os.remove(file_path)

        print(f'Processed case_id: {case_id}')

print('All cases processed.')
