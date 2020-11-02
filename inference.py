import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_inference', help='Path to file containing training annotations (see readme)', default='data/test_retinanet.csv')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)', default='data/train_retinanet.csv')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', default='data/class_retinanet.csv')

    parser.add_argument('--model_path', help='Path to file containing pretrained retinanet', default='csv_retinanet_68.pt')

    parser = parser.parse_args(args)


    dataset_inference = CSVDataset(train_file=parser.csv_inference, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_inference, batch_size=1, drop_last=False)
    dataloader_inference = DataLoader(dataset_inference, num_workers=3, collate_fn=collater, batch_sampler=sampler)


    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    use_gpu = True

    retinanet = torch.load(parser.model_path)

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    mAP = csv_eval.evaluate(dataset_inference, retinanet)
    ged = csv_eval.shap_eval(retinanet,dataset_train,dataset_inference)
    #print(mAP)

if __name__ == '__main__':
    main()
