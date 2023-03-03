import argparse
import datetime
import os
import sys
import os.path as osp
import time

import torch
from torch.backends import cudnn
from torch.optim import lr_scheduler

import models
from center_loss import CenterLoss
from dataset import Dataset, create_datasets
from imageaug import transform_for_training
from train import train, validate
from utils import Logger


def preparser():
    parser = argparse.ArgumentParser("Center Loss Example")
    # dataset
    parser.add_argument('--dataset_dir', type=str, default='datasets/stanford cars')
    parser.add_argument('--num_workers', default=4, type=int, help="number of data loading workers")
    # optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_model', type=float, default=0.001, help="learning rate for model")
    parser.add_argument('--lr_center', type=float, default=0.5, help="learning rate for center loss")
    parser.add_argument('--lambda_center', type=float, default=1, help="weight for center loss")
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
    # model
    parser.add_argument('--model', type=str, default='res18', choices=['res18', 'res50'])
    # misc
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--gpu', type=str, default='3,1,2,0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--image_shape', default=(128, 96), type=tuple)

    parser.add_argument('--plot', default=False, action='store_true', help="whether to plot features for every epoch")

    args = parser.parse_args()
    return args


def main():
    args = preparser()
    torch.manual_seed(args.seed)
    ##################################################
    args.plot = True
    args.dataset_dir = "datasets/stanford cars"
    # args.image_shape=(224,224)

    if sys.platform == 'win32':
        args.gpu = '0'
        args.num_workers = 1
    else:
        args.gpu = '3,1,2,0'
        args.num_workers = 4

    ##################################################
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset_dir + '.txt'))
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset_dir))
    training_set, validation_set, num_classes = create_datasets(args.dataset_dir)
    training_dataset = Dataset(
        training_set,
        transform_for_training(args.image_shape),
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True if use_gpu else False
    )
    validation_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True if use_gpu else False
    )

    print("Creating model: {}".format(args.model))
    model = models.create(
        name=args.model,
        num_classes=num_classes,
        image_shape=args.image_shape
    )
    feature_dim = model.FEATURE_DIM
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    criterion_cross_entropy = torch.nn.CrossEntropyLoss()
    criterion_center = CenterLoss(num_classes=num_classes, feature_dim=feature_dim, use_gpu=use_gpu)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=args.lr_center)

    if args.step_size > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.step_size, gamma=args.gamma)

    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        tacc, terr = train(model, criterion_cross_entropy, criterion_center,
                           optimizer_model, optimizer_center,
                           training_dataloader, use_gpu, num_classes, epoch, args)

        print("Accuracy (%): {}\t Error rate (%): {}".format(tacc, terr))

        if args.step_size > 0: scheduler.step()
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Validate")
            vacc, verr = validate(model, validation_dataloader, use_gpu, num_classes, epoch, args)
            print("Accuracy (%): {}\t Error rate (%): {}".format(vacc, verr))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    main()
    pass
