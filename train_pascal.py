import argparse
import collections

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader_pascal import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet.dataloader_style import StyleDataset
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='csv')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)',
                        default='data/train_Pascal_part.csv')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',
                        default='data/class_retinanet_Pascal.csv')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)',
                        default='data/val_Pascal_part.csv')

    parser.add_argument('--model_path', default='coco_resnet_50_map_0_335_state_dict.pt',
                        help='Path to file containing pretrained retinanet')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs_detection', help='Number of epochs for detection', type=int, default=50)
    parser.add_argument('--epochs_classification', help='Number of epochs for classification', type=int, default=50)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, out_classes=20)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, out_classes=20)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, out_classes=20)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, out_classes=20)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, out_classes=20)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if parser.model_path is not None:
        print('loading ', parser.model_path)
        if 'coco' in parser.model_path:
            retinanet.load_state_dict(torch.load(parser.model_path), strict=False)
        else:
            retinanet = torch.load(parser.model_path)
        print('Pretrained model loaded!')

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    # Here training the detection
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    loss_style_classif = nn.CrossEntropyLoss()

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    mAP_list = []
    mAPbest = 0
    for epoch_num in range(parser.epochs_detection):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    [classification_loss, regression_loss], style = retinanet(
                        [data['img'].cuda().float(), data['annot']])
                else:
                    [classification_loss, regression_loss], style = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                if torch.cuda.is_available():
                    style_loss = loss_style_classif(style, torch.tensor(data['style']).cuda())
                else:
                    style_loss = loss_style_classif(style, torch.tensor(data['style']))
                loss = classification_loss + regression_loss + style_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.4f} | Regression loss: {:1.4f} | Style loss: {:1.4f} | Running loss: {:1.4f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), float(style_loss),
                        np.mean(loss_hist)))

                del classification_loss
                del regression_loss
                del style_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            mAPclasses, mAP, accu = csv_eval.evaluate(dataset_val, retinanet)
            mAP_list.append(mAP)
            print('mAP_list', mAP_list)
        if mAP > mAPbest:
            print('Saving best checkpoint')
            torch.save(retinanet, 'model_best.pt')
            mAPbest = mAP

        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()
    torch.save(retinanet, 'model_final.pt')

    # Here we aggregate all the data to don't have to appy the Retinanet during training.
    retinanet.load_state_dict(torch.load('model_best.pt').state_dict())
    List_feature = []
    List_target = []
    retinanet.training = False
    retinanet.eval()
    retinanet.module.style_inference = True

    retinanet.module.freeze_bn()

    epoch_loss = []
    with torch.no_grad():
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    _, _, feature_vec = retinanet(data['img'].cuda().float())
                else:
                    _, _, feature_vec = retinanet(data['img'].float())
                List_feature.append(torch.squeeze(feature_vec).cpu())
                List_target.append(data['style'][0])
            except Exception as e:
                print(e)
                continue
    print('END of preparation of the data for classification of style')
    # Here begins Style training. Need to set to style_train. They are using the same loader, as it was expected to train both at the same time.

    batch_size_classification = 64
    dataloader_train_style = torch.utils.data.DataLoader(StyleDataset(List_feature, List_target),
                                                         batch_size=batch_size_classification)

    retinanet.load_state_dict(torch.load('model_best.pt').state_dict())

    # Here training the detection

    retinanet.module.style_inference = False
    retinanet.module.style_train(True)
    retinanet.training = True
    retinanet.train()
    optimizer = optim.Adam(retinanet.module.styleClassificationModel.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    loss_style_classif = nn.CrossEntropyLoss()
    retinanet.train()
    retinanet.module.freeze_bn()
    criterion = nn.CrossEntropyLoss()
    accu_list = []
    accubest = 0
    for epoch_num in range(parser.epochs_classification):

        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        total = 0
        correct = 0
        for iter_num, data in enumerate(dataloader_train_style):
            try:
                optimizer.zero_grad()
                inputs, targets = data
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = retinanet.module.styleClassificationModel(inputs, 0, 0, 0, True)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                total += targets.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum()

                print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                      % (epoch_num, parser.epochs_classification, iter_num + 1,
                         (len(dataloader_train_style) // batch_size_classification) + 1, loss.item(),
                         100. * correct / total))

            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            mAPclasses, mAP, accu = csv_eval.evaluate(dataset_val, retinanet)
            accu_list.append(accu)
            print('mAP_list', mAP_list, 'accu_list', accu_list)
        if accu > accubest:
            print('Saving best checkpoint')
            torch.save(retinanet.module, 'model_best_classif.pt')
            accubest = accu

        scheduler.step(accu)
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()
    torch.save(retinanet.module, 'model_final.pt')


if __name__ == '__main__':
    main()
