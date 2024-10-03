import os

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.recorder import Recorder

import torch
import numpy as np
import random
import time
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, random_split

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':  # SIngle sample cropped, no pre-loaded batches

    torch.cuda.empty_cache()
    opt = TrainOptions().parse()  # get training options
    dataset_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    full_dataset = dataset_loader.dataset

    dataset_size = len(full_dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    val_split = int(0.1 * dataset_size)
    train_split = dataset_size - val_split
    train_dataset, val_dataset = random_split(full_dataset, [train_split, val_split])

    print('The number of training images = %d' % train_split)
    print('The number of validation images = %d' % val_split)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    recorder = Recorder(opt)  # Recorder to monitor the progress

    scaler = amp.GradScaler()

    model.current_step = 0

    features_dir = os.path.join(opt.checkpoints_dir, opt.name, 'features')
    os.makedirs(features_dir, exist_ok=True)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

        model.set_current_epoch(epoch)
        epoch_loss = {}
        epoch_start_time = time.time()  # timer for entire epoch

        # training
        counter = 0
        for i, data in enumerate(train_loader):

            if opt.input_patch_size > 0:  # now can handle patches
                data_patched = {}
                patch_nums = len(data[next(iter(data))].squeeze(0))

                for j in range(0, patch_nums, opt.input_patch_size):
                    data_patched = {}
                    for key, value in data.items():
                        value = value.squeeze(0)
                        data_patched[key] = value[j % value.shape[0]:min((j % value.shape[0]) + opt.input_patch_size, patch_nums), ...]

                    model.set_input(data_patched)
                    with amp.autocast():
                        model.optimize_parameters()
                    losses = model.get_current_losses()

                    for k, v in losses.items():
                        epoch_loss[k] = epoch_loss[k] + v if k in epoch_loss else v

                    counter += 1
            else:
                model.set_input(data)
                with amp.autocast():
                    model.optimize_parameters()
                losses = model.get_current_losses()

                for k, v in losses.items():
                    epoch_loss[k] = epoch_loss.get(k, 0) + v

                counter += 1

        average_epoch_loss = {k: v / counter for k, v in epoch_loss.items()}

        accuracy = model.evaluate(val_loader)
        print(f'Epoch {epoch} Accuracy on validation set: {accuracy:.2f}%')

        recorder.print_epoch_losses(epoch, average_epoch_loss, time.time() - epoch_start_time)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d' % epoch)
            model.save_networks('latest')
            model.save_networks(epoch)

    model.save_stats()
    print('End of training')
    recorder.close()
