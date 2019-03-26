import torch
import time
import os
import sys
from tensorboardX import SummaryWriter
import pandas as pd
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

from birdsong.datasets.tools.sampling import upsample_df
from birdsong.datasets.tools.augmentation import ImageSoundscapeNoise
from birdsong.datasets.tools.enhancement import exponent
from birdsong.datasets.sequential import SpectralImageDataset
from birdsong.training import train, evaluate, logger, plot_conf_mat

if 'HOSTNAME' in os.environ:
    # script runs on server
    INPUT_DIR = '/storage/step1_slices/'
    TRAIN = pd.read_csv('mel_slices_train.csv')
    TEST = pd.read_csv('mel_slices_test.csv')
else:
    # script runs locally
    INPUT_DIR = 'storage/signal_images'
    TRAIN = pd.read_csv('top100_img_train.csv')
    TEST = pd.read_csv('top100_img_val.csv')


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN = torch.cuda.is_available()

print(f'Training on {DEVICE}')


def main(config_file):
    # read from config
    local_config = __import__(config_file)
    model_name = local_config.INPUTS['MODEL']
    model = getattr(__import__('birdsong.models',
                               fromlist=[model_name]), model_name)
    batch_size = local_config.INPUTS['BATCHSIZE']
    optimizer_name = local_config.INPUTS['OPTIMIZER']
    optimizer = getattr(__import__('torch.optim', fromlist=[optimizer_name]), optimizer_name)
    num_epochs = local_config.INPUTS['EPOCHS']
    no_classes = local_config.INPUTS['CLASSES']
    learning_rate = local_config.INPUTS['LR']

    # logging
    start_time = time.time()
    date = time.strftime('%d-%m-%Y-%H-%M-%S', time.localtime())
    log_path = f'./birdsong/run_log/{model_name}_{date}'
    state_fname, log_fname, summ_tensor_board = logger.create_log(log_path)
    writer = SummaryWriter(str(summ_tensor_board))

    # Augmentation
    noiser = ImageSoundscapeNoise('storage/noise_images', scaling=0.3)

    ds_train = SpectralImageDataset(
        TRAIN, INPUT_DIR, enhancement_func=exponent, augmentation_func=noiser)
    ds_test = SpectralImageDataset(TEST, INPUT_DIR, enhancement_func=exponent)

    dl_train = DataLoader(ds_train, batch_size,
                          num_workers=4, pin_memory=PIN, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size, num_workers=4,
                         pin_memory=PIN, shuffle=True)
    print('Dataloaders initialized')

    time_axis = ds_test.shape[1]
    freq_axis = ds_test.shape[0]
    net = model(time_axis=time_axis, freq_axis=freq_axis,
                no_classes=no_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net.parameters(), lr=learning_rate)

    # local vars
    best_acc = 0
    for epoch in range(num_epochs):
        train(net, dl_train, epoch, optimizer, criterion, DEVICE)

        train_stats = evaluate(
            net, dl_train, criterion, no_classes, DEVICE)
        print(
            f'Training: Loss: {train_stats[0]:.5f}, Acc: {train_stats[1]:.5f}, Top 5: {train_stats[2]:.5f}')

        test_stats = evaluate(
            net, dl_test, criterion, no_classes, DEVICE)
        print(
            f'Validation: Loss: {test_stats[0]:.5f}, Acc: {test_stats[1]:.5f}, Top 5: {test_stats[2]:.5f}')

        is_best = test_stats[1] > best_acc
        best_acc = max(test_stats[1], best_acc)
        print('Best Accuracy: {:.5f}'.format(best_acc))

        logger.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_acc,
            #'label_dict': ds_train.encoder.codes,
            # 'model': net,
        }, is_best, filename=state_fname)

        """
        print('Making images')
        img_path = log_path + '/train' + '_' + str(epoch) + '.png'
        img = plot_conf_mat(img_path, train_conf_matrix)

        img_path = log_path + '/test' + '_' + str(epoch) + '.png'
        img = plot_conf_mat(img_path, test_conf_matrix)
        """

        logger.write_summary(writer, epoch, train_stats, test_stats)
        logger.dump_log_txt(date, start_time, local_config,
                            train_stats, test_stats, best_acc, log_fname)

    writer.close()
    print('Finished Training')

    with open(os.path.join(log_path, 'model.pt'), 'wb') as f:
        torch.save(net, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('usage: %s config_file' % os.path.basename(sys.argv[0]))
        sys.exit(2)

    CONFIG = os.path.basename(sys.argv[1])

    if CONFIG[-3:] == ".py":
        CONFIG = CONFIG[:-3]

    main(CONFIG)
