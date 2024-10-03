from abc import ABC, abstractmethod
import numpy as np
from utils.patch_operations import *


class BaseProcessing(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain

    @abstractmethod
    def preprocessing(self, sample):
        pass

    @abstractmethod
    def postprocessing(self, preprocessed_sample):
        pass

    def get_norm(self):
        norm = getattr(self, self.opt + '_norm')
        return norm

    def instance_norm_3D(self, image, mask=None):
        mask = mask if mask is not None else np.ones(image.shape)
        means = np.mean(image[mask == 1], axis=0)
        stds = np.std(image[mask == 1], axis=0)

        image[mask == 1] = (image[mask == 1] - means) / stds
        image[mask != 1] = 0

        return image, means, stds

    def minmax_norm_3D(self, image, mask=None):
        mask = mask if mask is not None else np.ones(image.shape)
        max_val = np.max(image[mask == 1])
        min_val = np.min(image[mask == 1])

        image[mask == 1] = (image[mask == 1] - min_val) / (max_val - min_val)
        image[mask != 1] = 0

        return image, max_val, min_val

    def layer_norm_3D(self, image, mask=None):
        mask = mask if mask is not None else np.ones(image.shape)
        means = np.mean(image[mask == 1])
        stds = np.std(image[mask == 1])

        image[mask == 1] = (image[mask == 1] - means) / stds
        image[mask != 1] = 0

        return image, means, stds

    def find_bounding_box_3D(self, mask):
        x, y, z = mask.shape[0], mask.shape[1], mask.shape[2]

        for i in range(z):
            slice = mask[:, :, i]
            if np.sum(slice) > 0:
                save_z_from_I = i
                break

        for i in reversed(range(z)):
            slice = mask[:, :, i]
            if np.sum(slice) > 0:
                save_z_from_S = i
                break

        for i in range(y):
            slice = mask[:, i, :]
            if np.sum(slice) > 0:
                save_y_from_P = i
                break

        for i in reversed(range(y)):
            slice = mask[:, i, :]
            if np.sum(slice) > 0:
                save_y_from_A = i
                break

        for i in range(x):
            slice = mask[i, :, :]
            if np.sum(slice) > 0:
                save_x_from_L = i
                break

        for i in reversed(range(x)):
            slice = mask[i, :, :]
            if np.sum(slice) > 0:
                save_x_from_R = i
                break

        return save_x_from_L, save_x_from_R, save_y_from_P, save_y_from_A, save_z_from_I, save_z_from_S

    def patch_generation(self, image, kernel_size, stride, three_dim, index=None):
        patches, coords = slice_matrix(image, kernel_size, stride, three_dim, index=index, save_coords=True)

        if self.isTrain:
            for i in reversed(range(0, len(patches))):
                if np.sum(patches[i]) == 0:
                    del patches[i]
                    del coords[i]

        patches = np.stack(patches, axis=0)
        coords = np.stack(coords, axis=0)

        return patches, coords

    def patch_generation_by_coords(self, image, coords):
        patches = []
        for patch_coord in coords:
            x_start = patch_coord['x_start']
            x_end = patch_coord['x_end']
            y_start = patch_coord['y_start']
            y_end = patch_coord['y_end']
            z_start = patch_coord['z_start']
            z_end = patch_coord['z_end']

            patch = image[x_start:x_end, y_start:y_end, z_start:z_end]
            patches.append(patch)

        patches = np.stack(patches, axis=0)
        return patches
