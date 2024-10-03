from data.base_dataset import BaseDataset
from data.io import create_io
from data.processing import create_processing
from utils.dmri_io import get_HCPsamples, get_HCPEvalsamples
import os
import torch
import numpy as np
import pickle


class HCPUADDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.artifact_to_label = {"bias": 0, "corrupted_volume": 1, "limited_field_of_view": 2}

        list_path = "/home/sheng/used_by_zhenlin/UdAD_supervised"
        if self.opt.isTrain:
            hcp_split_path = os.path.join(list_path, 'HCP_train_list.pickle')
            with open(hcp_split_path, 'rb') as f:
                self.sample_dict = pickle.load(f)

            self.sample_list = []
            for label, sample_ids in self.sample_dict.items():
                label = self.artifact_to_label[label]
                for sample_id in sample_ids:
                    self.sample_list.append((sample_id, label))
        else:
            hcp_split_path = os.path.join(list_path, 'HCP_eval_list.pickle')
            self.sample_list = get_HCPEvalsamples(hcp_split_path)

        self.io = create_io('hcpUAD', opt)

        self.processing = create_processing('UAD', opt)

    def __len__(self):
        dataset_length = len(self.sample_list)
        return dataset_length

    def __getitem__(self, index):
        sample_index, label = self.sample_list[index]
        sample = self.io.load_sample(sample_index)
        self.processing.preprocessing(sample)

        print(f"Sample ID: {sample_index}, Label: {label}")

        if not self.opt.isTrain:
            self.preprocessed_sample = sample

        if self.opt.input_patch_size > 0:
            b0 = torch.from_numpy(sample.b0_processed).unsqueeze(0).permute(1, 0, 2, 3, 4)  # (1, w, h, d)
            dwis = torch.from_numpy(sample.dwis_processed).permute(0, 4, 1, 2, 3)  # (c, w, h, d)
            bm = torch.from_numpy(sample.bm_processed).unsqueeze(0).permute(1, 0, 2, 3, 4)  # (1, w, h, d)
            dti = torch.from_numpy(sample.dti_processed).permute(0, 4, 1, 2, 3)  # (c, w, h, d)
            fa = torch.from_numpy(sample.fa_processed).unsqueeze(0).permute(1, 0, 2, 3, 4)  # (1, w, h, d)
        else:
            b0 = torch.from_numpy(sample.b0_processed).unsqueeze(0)  # (1, w, h, d)
            dwis = torch.from_numpy(sample.dwis_processed).permute(3, 0, 1, 2)  # (c, w, h, d)
            bm = torch.from_numpy(sample.bm_processed).unsqueeze(0)  # (1, w, h, d)
            dti = torch.from_numpy(sample.dti_processed).permute(3, 0, 1, 2)  # (c, w, h, d)
            fa = torch.from_numpy(sample.fa_processed).unsqueeze(0)  # (1, w, h, d)

        label = torch.tensor(label, dtype=torch.long)

        items = {
            'b0': b0,
            'dwis': dwis,
            'dti': dti,
            'fa': fa,
            'bm': bm,
            'id': sample_index,
            'label': label
        }

        return items

    def postprocessing(self, outputs, counter):
        self.preprocessed_sample.out_dict = outputs
        resulted_dict = self.processing.postprocessing(self.preprocessed_sample)

        if counter % self.opt.save_prediction == 0:
            for key in resulted_dict.keys():
                name = key
                self.io.save_sample((resulted_dict[key], self.preprocessed_sample.affine, self.preprocessed_sample.index, name))
