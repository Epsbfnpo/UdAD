import numpy as np
import math


def crop_patch(patch, slicer):
    """
    Crops a 3D patch based on the provided slicer and rearranges its axes.

    This function transforms the patch from channel-last to channel-first structure, crops the patch according to the specified slicer, and then transforms it back to channel-last structure.

    Args:
        patch (np.ndarray): The 3D patch to be cropped, with the last axis representing channels.
        slicer (list of slice): A list of slice objects defining how to crop the patch.

    Returns:
        np.ndarray: The cropped patch with the same structure as the input.
    """
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Exclude the number of batches and classes from the slice range
    slicer[0] = slice(None)
    slicer[1] = slice(None)
    # Crop patches according to slicer
    patch_cropped = patch[tuple(slicer)]
    # Transform data from channel-first back to channel-last structure
    patch_cropped = np.moveaxis(patch_cropped, 1, -1)
    # Return cropped patch
    return patch_cropped


# Slice a matrix
def slice_matrix(array, window, overlap, three_dim, index=None, save_coords=False):
    """
    Slices a 2D or 3D matrix into smaller patches based on the specified window and overlap.

    This function determines whether to slice a 2D or 3D matrix and delegates to the appropriate slicing function.

    Args:
        array (np.ndarray): The matrix to be sliced.
        window (tuple of int): The size of the window (patch) to slice.
        overlap (tuple of int): The overlap between adjacent patches.
        three_dim (bool): If True, the matrix is 3D; otherwise, it is 2D.
        index (int, optional): An index to identify the patch. Default is None.
        save_coords (bool, optional): If True, saves the coordinates of each patch. Default is False.

    Returns:
        list: A list of patches (and optionally their coordinates) extracted from the matrix.
    """
    if three_dim:
        return slice_3Dmatrix(array, window, overlap, index, save_coords)
    else:
        return slice_2Dmatrix(array, window, overlap)


# Concatenate a matrix
def concat_matrices(patches, image_size, window, overlap, three_dim, coords=None):
    """
    Concatenates a list of patches into a 2D or 3D matrix.

    This function determines whether to concatenate 2D or 3D patches and delegates to the appropriate concatenation function.

    Args:
        patches (list of np.ndarray): The list of patches to be concatenated.
        image_size (tuple of int): The size of the original image/matrix.
        window (tuple of int): The size of the window (patch).
        overlap (tuple of int): The overlap between adjacent patches.
        three_dim (bool): If True, the matrix is 3D; otherwise, it is 2D.
        coords (list of dict, optional): The coordinates of the patches, if applicable. Default is None.

    Returns:
        np.ndarray: The concatenated matrix.
    """
    if three_dim:
        return concat_3Dmatrices(patches, image_size, window, overlap, coords)
    else:
        return concat_2Dmatrices(patches, image_size, window, overlap)


# Slice a 2D matrix
def slice_2Dmatrix(array, window, overlap):
    """
    Slices a 2D matrix into smaller patches based on the specified window and overlap.

    Args:
        array (np.ndarray): The 2D matrix to be sliced.
        window (tuple of int): The size of the window (patch) to slice.
        overlap (tuple of int): The overlap between adjacent patches.

    Returns:
        list of np.ndarray: A list of patches extracted from the matrix.
    """
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) / float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) / float(window[1] - overlap[1])))

    # Iterate over it x,y
    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):

            # Define window edges
            x_start = x*window[0] - x*overlap[0]
            x_end = x_start + window[0]
            y_start = y*window[1] - y*overlap[1]
            y_end = y_start + window[1]
            # Adjust ends
            if x_end > len(array):
                # Create an overlapping patch for the last images / edges
                # to ensure the fixed patch/window sizes
                x_start = len(array) - window[0]
                x_end = len(array)
                # Fix for MRIs which are smaller than patch size
                if x_start < 0:
                    x_start = 0
            if y_end > len(array[0]):
                y_start = len(array[0]) - window[1]
                y_end = len(array[0])
                # Fix for MRIs which are smaller than patch size
                if y_start < 0:
                    y_start = 0
            # Cut window
            window_cut = array[x_start:x_end, y_start:y_end]
            # Add to result list
            patches.append(window_cut)
    return patches


# Concatenate a list of patches together to a numpy matrix
def concat_2Dmatrices(patches, image_size, window, overlap):
    """
    Concatenates a list of 2D patches into a single 2D matrix.

    Args:
        patches (list of np.ndarray): The list of 2D patches to be concatenated.
        image_size (tuple of int): The size of the original 2D image/matrix.
        window (tuple of int): The size of the window (patch).
        overlap (tuple of int): The overlap between adjacent patches.

    Returns:
        np.ndarray: The concatenated 2D matrix.
    """
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) / float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) / float(window[1] - overlap[1])))

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            # Calculate pointer from 2D steps to 1D list of patches
            pointer = x*steps_y + y
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = patches[pointer]
            else:
                matrix_p = patches[pointer]
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap, image_size, window, 1)
                matrix_y, matrix_p = handle_overlap(matrix_y, matrix_p, slice_overlap, axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_p), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap, image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y, slice_overlap, axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return matrix_x


# Slice a 3D matrix
def slice_3Dmatrix(array, window, overlap, index=None, save_coords=False):
    """
    Slices a 3D matrix into smaller patches based on the specified window and overlap.

    Args:
        array (np.ndarray): The 3D matrix to be sliced.
        window (tuple of int): The size of the window (patch) to slice.
        overlap (tuple of int): The overlap between adjacent patches.
        index (int, optional): An index to identify the patch. Default is None.
        save_coords (bool, optional): If True, saves the coordinates of each patch. Default is False.

    Returns:
        list of np.ndarray or tuple: A list of patches (and optionally their coordinates) extracted from the matrix.
    """
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) / float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) / float(window[1] - overlap[1])))
    steps_z = int(math.ceil((len(array[0][0]) - overlap[2]) / float(window[2] - overlap[2])))

    # print('slice3d', array.shape)
    # Iterate over it x,y,z
    patches = []
    coords = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x*window[0] - x*overlap[0]
                x_end = x_start + window[0]
                y_start = y*window[1] - y*overlap[1]
                y_end = y_start + window[1]
                z_start = z*window[2] - z*overlap[2]
                z_end = z_start + window[2]
                # Adjust ends
                if x_end > len(array):
                    # Create an overlapping patch for the last images / edges
                    # to ensure the fixed patch/window sizes
                    x_start = len(array) - window[0]
                    x_end = len(array)
                    # Fix for MRIs which are smaller than patch size
                    if x_start < 0:
                        x_start = 0
                if y_end > len(array[0]):
                    y_start = len(array[0]) - window[1]
                    y_end = len(array[0])
                    # Fix for MRIs which are smaller than patch size
                    if y_start < 0:
                        y_start = 0
                if z_end > len(array[0][0]):
                    z_start = len(array[0][0]) - window[2]
                    z_end = len(array[0][0])
                    # Fix for MRIs which are smaller than patch size
                    if z_start < 0:
                        z_start = 0
                # Cut window
                window_cut = array[x_start:x_end, y_start:y_end, z_start:z_end]
                # Add to result list
                patches.append(window_cut)

                coord_dict = {
                    'index': index,
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_start': y_start,
                    'y_end': y_end,
                    'z_start': z_start,
                    'z_end': z_end
                }
                coords.append(coord_dict)

    if not save_coords:
        return patches
    else:
        return patches, coords


# Concatenate a list of patches together to a numpy matrix
def concat_3Dmatrices(patches, image_size, window, overlap, coords=None):
    """
    Concatenates a list of 3D patches into a single 3D matrix.

    Args:
        patches (list of np.ndarray): The list of 3D patches to be concatenated.
        image_size (tuple of int): The size of the original 3D image/matrix.
        window (tuple of int): The size of the window (patch).
        overlap (tuple of int): The overlap between adjacent patches.
        coords (list of dict, optional): The coordinates of the patches, if applicable. Default is None.

    Returns:
        np.ndarray: The concatenated 3D matrix.
    """
    if coords is None:
        # Calculate steps
        steps_x = int(math.ceil((image_size[0] - overlap[0]) / float(window[0] - overlap[0])))
        steps_y = int(math.ceil((image_size[1] - overlap[1]) / float(window[1] - overlap[1])))
        steps_z = int(math.ceil((image_size[2] - overlap[2]) / float(window[2] - overlap[2])))

        # print('steps_z', steps_z)

        # Iterate over it x,y,z
        matrix_x = None
        matrix_y = None
        matrix_z = None
        pointer = 0

        counts = np.zeros((steps_x, steps_y, steps_z))
        # print('counts.shape', counts.shape)
        # print('patches', patches.shape)
        for x in range(0, steps_x):
            for y in range(0, steps_y):
                for z in range(0, steps_z):
                    # Calculate pointer from 3D steps to 1D list of patches
                    pointer = z + y*steps_z + x*steps_y*steps_z
                    # Connect current patch to temporary Matrix Z
                    if z == 0:
                        matrix_z = patches[pointer]
                    else:
                        matrix_p = patches[pointer]
                        # print('matrix_p', matrix_p.shape)
                        # Handle z-axis overlap
                        # print('z',x, y, z)
                        counts[x, y, z] += 1
                        slice_overlap = calculate_overlap(z, steps_z, overlap, image_size, window, 2)
                        matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p, slice_overlap, axis=2)
                        matrix_z = np.concatenate((matrix_z, matrix_p), axis=2)
                # Connect current tmp Matrix Z to tmp Matrix Y
                if y == 0:
                    matrix_y = matrix_z
                else:
                    # Handle y-axis overlap
                    # print('y', x, y, z)
                    counts[x, y, z] += 1
                    slice_overlap = calculate_overlap(y, steps_y, overlap, image_size, window, 1)
                    matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z, slice_overlap, axis=1)
                    matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
            # Connect current tmp Matrix Y to final Matrix X
            if x == 0:
                matrix_x = matrix_y
            else:
                # Handle x-axis overlap
                # print('x', x, y, z)
                counts[x, y, z] += 1
                slice_overlap = calculate_overlap(x, steps_x, overlap, image_size, window, 0)
                matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y, slice_overlap, axis=0)
                matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
        # Return final combined matrix
        return matrix_x
    else:
        img = np.zeros(image_size, dtype=np.float32)
        counts = np.zeros(image_size, dtype=np.float32)
        for ci, coord in enumerate(coords):
            sub_index = coord['index']
            x_start = coord['x_start']
            x_end = coord['x_end']
            y_start = coord['y_start']
            y_end = coord['y_end']
            z_start = coord['z_start']
            z_end = coord['z_end']
            patch = patches[ci, :, :, :]

            img[x_start:x_end, y_start:y_end, z_start:z_end] += patch
            counts[x_start:x_end, y_start:y_end, z_start:z_end] += 1

        counts[counts == 0] = 1
        img /= counts
        return img


# Calculate the overlap of the current matrix slice
def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
    """
    Calculates the overlap of the current matrix slice based on its position.

    Args:
        pointer (int): The current position in the matrix.
        steps (int): The total number of steps along the axis.
        overlap (tuple of int): The overlap between adjacent patches.
        image_size (tuple of int): The size of the original image/matrix.
        window (tuple of int): The size of the window (patch).
        axis (int): The axis along which the overlap is calculated.

    Returns:
        int: The calculated overlap size for the current slice.
    """
    # Overlap: IF last axis-layer -> use special overlap size
    if pointer == steps-1 and not (image_size[axis]-overlap[axis]) % (window[axis]-overlap[axis]) == 0:
        current_overlap = window[axis] - (image_size[axis] - overlap[axis]) % (window[axis] - overlap[axis])
    # Overlap: ELSE -> use default overlap size
    else:
        current_overlap = overlap[axis]
    # Return overlap
    return current_overlap


# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis):
    """
    Handles the overlap between two overlapping matrices by averaging the overlapping region.

    Args:
        matrixA (np.ndarray): The first matrix (existing region).
        matrixB (np.ndarray): The second matrix (new region).
        overlap (int): The size of the overlap between the matrices.
        axis (int): The axis along which the overlap occurs.

    Returns:
        tuple of np.ndarray: The processed matrices with the overlap handled.
    """
    # Access overllaping slice from matrix A
    idxA = [slice(None)] * matrixA.ndim
    matrixA_shape = matrixA.shape
    idxA[axis] = range(matrixA_shape[axis] - overlap, matrixA_shape[axis])
    sliceA = matrixA[tuple(idxA)]
    # Access overllaping slice from matrix B
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    # Calculate Average prediction values between the two matrices
    # and save them in matrix A
    matrixA[tuple(idxA)] = np.mean(np.array([sliceA, sliceB]), axis=0)
    # Remove overlap from matrix B
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB


def pad_3d_image(image, target_shape, padding_value=0):
    """
    Pads a 3D image to the target shape with the specified padding value.

    Args:
        image (np.ndarray): The 3D image to be padded.
        target_shape (tuple of int): The target shape of the padded image.
        padding_value (int, optional): The value used for padding. Default is 0.

    Returns:
        np.ndarray: The padded 3D image.
    """
    current_shape = image.shape
    pad_width = []

    for current_dim, target_dim in zip(current_shape, target_shape):
        if current_dim < target_dim:
            total_pad = target_dim - current_dim
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=padding_value)
    return padded_image


def unpad_3d_image(padded_image, original_shape):
    """
    Removes padding from a 3D image to return it to its original shape.

    Args:
        padded_image (np.ndarray): The padded 3D image.
        original_shape (tuple of int): The original shape before padding.

    Returns:
        np.ndarray: The unpadded 3D image.
    """
    w, h, d, c = original_shape
    current_shape = padded_image.shape

    # Calculate the amount of padding to remove
    pad_w_before = (current_shape[0] - w) // 2
    pad_h_before = (current_shape[1] - h) // 2
    pad_d_before = (current_shape[2] - d) // 2

    unpadded_image = padded_image[pad_w_before:pad_w_before+w, pad_h_before:pad_h_before+h, pad_d_before:pad_d_before+d, :]
    return unpadded_image
