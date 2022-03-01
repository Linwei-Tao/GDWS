##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import json
import sys, time, random, argparse
from copy import deepcopy
import torch.nn.functional as F

import numpy as np
import torch
from datasets import get_datasets, get_search_loaders
from utils import prepare_seed, dict2obj, AverageMeter, expected_calibration_error, test_classification_net, \
    get_soft_binning_ece_tensor
from models.model_utils import gwdsresnet50


def search_func(
        valid_loader,
        test_loader,
        network,
        criterion,
        a_optimizer,
        soft_ece=False
):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    for step, (inputs, targets) in enumerate(valid_loader):

        # if step % 50 == 0:
        #     print(network.get_arch())
        # acc, ece, test_loss = evaluate(test_loader, network)
        # print("acc: {}, ece: {}, test loss: {}, train loss: {}".format(acc, ece, test_loss, losses.avg))
        # network.load_back()

        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        # update the architecture-weight
        a_optimizer.zero_grad()
        logits = network(inputs)
        ce_loss = criterion(logits, targets)
        confidence_vals, predictions = torch.max(logits, dim=1)
        if soft_ece:
            soft_ece = get_soft_binning_ece_tensor(predictions,
                                                   targets,
                                                   soft_binning_bins=15,
                                                   soft_binning_use_decay=True,
                                                   soft_binning_decay_factor=1e-4,
                                                   soft_binning_temp=1)
            loss = ce_loss + soft_ece
        else:
            loss = ce_loss
        loss.backward()
        a_optimizer.step()

        losses.update(loss.item(), inputs.size(0))

    return losses.avg


def evaluate(test_loader, network, load_weight=True):
    device = torch.device("cuda")

    if load_weight:
        network.load_weight()

    test_conf_matrix, test_acc, test_labels, test_predictions, test_confidences, test_loss = test_classification_net(
        network,
        test_loader,
        device)

    test_ece = expected_calibration_error(test_confidences, test_predictions, test_labels, num_bins=15)
    network.load_back()
    return test_acc, test_ece, test_loss


def main(xargs):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    train_data, test_data, xshape, class_num = get_datasets(
        name=xargs.dataset, root=xargs.data_path, cutout=-1
    )

    config_file = open("./configs/config.json")
    config_dict = json.load(config_file)
    config = dict2obj(config_dict)

    search_loader, train_loader, valid_loader, test_loader = get_search_loaders(
        train_data,
        test_data,
        xargs.dataset,
        "configs/dataset-split/",
        config.batch_size,
        xargs.workers,
    )

    # build backbone
    search_model = gwdsresnet50()

    criterion = torch.nn.CrossEntropyLoss()

    a_optimizer = torch.optim.Adam(
        search_model.get_alphas(),
        lr=xargs.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=xargs.arch_weight_decay,
    )

    # a_optimizer = torch.optim.SGD(
    #     search_model.get_alphas(),
    #     lr=xargs.arch_learning_rate,
    #     weight_decay=xargs.arch_weight_decay,
    # )

    network, criterion = search_model.cuda(), criterion.cuda()

    # start training
    total_epoch = 1500
    network.load_baseline()

    for epoch in range(total_epoch):
        search_model.set_tau(
            xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1)
        )
        # print(network.layer1.state_dict()['0.conv1.weight'])

        a_loss = search_func(
            valid_loader,
            test_loader,
            network,
            criterion,
            a_optimizer,
            soft_ece=False
        )

        # print(network.layer1.state_dict()['0.conv1.weight'])

        acc, ece, loss = evaluate(test_loader, network, load_weight=True)

        print("Epoch {} ==> arch: {}, acc: {}, ece: {}, test loss: {}, arch loss: {}, tau: {}".format(epoch + 1,
                                                                                                      network.get_arch().cpu().numpy(),
                                                                                                      acc, ece, loss,
                                                                                                      a_loss,
                                                                                                      network.tau))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GDWS")
    parser.add_argument("--data_path", type=str, help="The path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--sample_strategy", type=str, help="The search space sampling strategy.")
    # architecture leraning rate
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=3e-4,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument("--tau_min", type=float, help="The minimum tau for Gumbel")
    parser.add_argument("--tau_max", type=float, help="The maximum tau for Gumbel")
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log.", default="temp_dir"
    )
    parser.add_argument("--print_freq", type=int, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", type=int, help="manual seed", default=0)
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
