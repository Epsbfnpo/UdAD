import os
import nibabel as nib
import pickle
from utils.mrtrix import *


def load_data(path, needs_affine=False):
    """
    Loads imaging data from a given path.

    Args:
        path (str): The file path to the data.
        needs_affine (bool): Whether to return the affine matrix along with the data.

    Returns:
        data_copied (numpy.ndarray): The imaging data.
        affine_copied (numpy.ndarray, optional): The affine matrix (if needs_affine=True).
    """
    if not os.path.exists(path):
        raise ValueError("Data could not be found \"{}\"".format(path))

    if path.endswith('.mif.gz') or path.endswith('.mif'):
        vol = load_mrtrix(path)
        data_copied = vol.data.copy()
        affine_copied = vol.transform.copy()
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        vol = nib.load(path)
        data_copied = vol.get_fdata().copy()
        affine_copied = vol.affine.copy()
    else:
        raise IOError('file extension not supported: ' + str(path))

    # Return volume
    if needs_affine:
        return data_copied, affine_copied
    else:
        return data_copied


def load_random6DWIs(path, needs_affine=False):
    """
    Loads a random selection of 6 DWIs from the provided path.

    Args:
        path (str): The file path to the data.
        needs_affine (bool): Whether to return the affine matrix along with the data.

    Returns:
        b0 (numpy.ndarray): The b0 image.
        random6_dwis (numpy.ndarray): A random selection of 6 DWIs.
        affine_copied (numpy.ndarray, optional): The affine matrix (if needs_affine=True).
    """
    if not os.path.exists(path):
        raise ValueError("Data could not be found \"{}\"".format(path))

    if path.endswith('.mif.gz') or path.endswith('.mif'):
        vol = load_mrtrix(path)
        data_copied = vol.data.copy()
        affine_copied = vol.transform.copy()
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        vol = nib.load(path)
        data_copied = vol.get_fdata().copy()
        affine_copied = vol.affine.copy()
    else:
        raise IOError('file extension not supported: ' + str(path))

    b0 = data_copied[..., 0]
    random6_index = [1, 2, 3, 4, 5, 6]
    random6_dwis = data_copied[..., random6_index]

    return b0, random6_dwis, affine_copied


def save_data(data, affine, output_name):
    """
    Saves imaging data to a NIfTI file.

    Args:
        data (numpy.ndarray): The data to save.
        affine (numpy.ndarray): The affine matrix associated with the data.
        output_name (str): The file path to save the data.
    """
    nifti = nib.Nifti1Image(data, affine=affine)
    nib.save(nifti, output_name)


def get_HCPsamples(hcp_split_path, train=True):
    """
    Retrieves the list of HCP samples from a pickle file.

    Args:
        hcp_split_path (str): The path to the pickle file containing the HCP samples.
        train (bool): Whether to retrieve the training samples or test samples.

    Returns:
        sample_list (list): The list of samples.
    """
    if not os.path.exists(hcp_split_path):
        raise IOError("hcp splited list path, {}, could not be resolved".format(hcp_split_path))

    with open(hcp_split_path, 'rb') as handle:
        sub_list = pickle.load(handle)
        
        if train:
            sample_list = sub_list['train']
        else:
            sample_list = sub_list['test']

    return sample_list


def get_HCPEvalsamples(hcp_eval_path):
    """
    Retrieves the list of HCP evaluation samples from a pickle file.

    Args:
        hcp_eval_path (str): The path to the pickle file containing the HCP evaluation samples.

    Returns:
        sample_list (list): The list of evaluation samples.
    """
    if not os.path.exists(hcp_eval_path):
        raise IOError("hcp eval splited list path, {}, could not be resolved".format(hcp_eval_path))

    with open(hcp_eval_path, 'rb') as handle:
        sub_list = pickle.load(handle)
        
        sample_list = []
        sample_list.extend(sub_list['normal'])
        sample_list.extend(sub_list['abnormal'])

    return sample_list


def save_pickle(file, path):
    """
    Saves a Python object to a pickle file.

    Args:
        file (object): The Python object to save.
        path (str): The file path to save the pickle file.
    """
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    """
    Reads a Python object from a pickle file.

    Args:
        path (str): The file path of the pickle file to read.

    Returns:
        pickle_file (object): The Python object loaded from the pickle file.
    """
    with open(path, 'rb') as handle:
        pickle_file = pickle.load(handle)
    return pickle_file


def list_dir(directory):
    """
    Lists all non-hidden files in a directory.

    Args:
        directory (str): The directory to list files from.

    Returns:
        visible_files (list): A list of non-hidden files in the directory.
    """
    visible_files = []
    for file in os.listdir(directory):
        if not file.startswith('.'):
            visible_files.append(file)
    return visible_files
