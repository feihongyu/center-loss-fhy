import numpy
import torch

from utils import AverageMeter, plot_features


def train(model, criterion_cross_entropy, criterion_center,
          optimizer_model, optimizer_center,
          training_dataloader, use_gpu, num_classes, epoch, args):
    model.train()
    cross_entropy_losses = AverageMeter()
    center_losses = AverageMeter()
    losses = AverageMeter()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    for batch_index, (data, labels, names) in enumerate(training_dataloader):
        if use_gpu:
            data = data.cuda()
            labels = labels.cuda()

        outputs, features = model(data)

        predictions = outputs.data.max(1)[1]
        total += labels.size(0)
        correct += (predictions == labels.data).sum()

        loss_cross_entropy = criterion_cross_entropy(outputs, labels)
        loss_center = criterion_center(features, labels)
        loss = loss_cross_entropy + args.lambda_center * loss_center

        optimizer_model.zero_grad()
        optimizer_center.zero_grad()

        loss.backward()
        # or
        # loss_cross_entropy.backward()
        # loss_center.backward()

        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        # for param in criterion_cent.parameters():
        #     param.grad.data *= (1. / args.weight_cent)
        optimizer_center.step()

        losses.update(loss.item(), labels.size(0))
        cross_entropy_losses.update(loss_cross_entropy.item(), labels.size(0))
        center_losses.update(loss_center.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features[:, 0:2].data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features[:, 0:2].data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_index + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})".format(
                batch_index + 1, len(training_dataloader),
                losses.val, losses.avg,
                cross_entropy_losses.val, cross_entropy_losses.avg,
                center_losses.val, center_losses.avg
            )
            )

    if args.plot:
        all_features = numpy.concatenate(all_features, 0)
        all_labels = numpy.concatenate(all_labels, 0)
        numpy.savetxt(str(args.save_dir) + "/" + str(epoch) + "train.txt",
                      numpy.concatenate((all_features, all_labels[:, None]), axis=-1))
        # plot_features(all_features, all_labels, num_classes, epoch, 'train', args.save_dir)

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def validate(model, validation_dataloader, use_gpu, num_classes, epoch, args):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels, names in validation_dataloader:
            if use_gpu:
                data = data.cuda()
                labels = labels.cuda()

            outputs, features = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if args.plot:
                if use_gpu:
                    all_features.append(features[:, 0:2].data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features[:, 0:2].data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = numpy.concatenate(all_features, 0)
        all_labels = numpy.concatenate(all_labels, 0)
        numpy.savetxt(str(args.save_dir) + "/" + str(epoch) + "validate.txt",
                      numpy.concatenate((all_features,  all_labels[:, None]), axis=-1))
        # plot_features(all_features, all_labels, num_classes, epoch, 'test', args.save_dir)

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err
