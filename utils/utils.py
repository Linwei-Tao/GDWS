import math
import os, sys, torch, random, PIL, copy, numpy as np
from collections import namedtuple
from os import path as osp
from shutil import copyfile
# import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from .logger import AverageMeter

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
    args = copy.deepcopy(xargs)
    from utils import Logger

    logger = Logger(args.save_dir, args.rand_seed)
    logger.log("Main Function with logger : {:}".format(logger))
    logger.log("Arguments : -------------------------------")
    for name, value in args._get_kwargs():
        logger.log("{:16} : {:}".format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    return logger


def dict2obj(dict):
    return namedtuple("ObjectName", dict.keys())(*dict.values())


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def time_string():
    ISOTIMEFORMAT = "%Y-%m-%d %X"
    string = "[{:}]".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def expected_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
               abs(bin_accuracy - bin_confidence)
    return ece


def test_classification_net(model, data_loader, device):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    logits_list = []
    predictions_list = []
    confidence_vals_list = []
    losses = AverageMeter()
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)

            logits = model.test_forward(data)
            loss = F.cross_entropy(logits, label)
            softmax = F.softmax(logits, dim=1)
            confidence_vals, predictions = torch.max(softmax, dim=1)
            losses.update(loss.item(), data.size(0))

            logits_list.extend(softmax.cpu().numpy().tolist())
            labels_list.extend(label.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list, \
           predictions_list, confidence_vals_list, losses.avg


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
                              (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                                       float(bin_dict[binn][COUNT])
    return bin_dict


def get_soft_binning_ece_tensor(predictions,
                                labels,
                                soft_binning_bins=15,
                                soft_binning_use_decay=True,
                                soft_binning_decay_factor=1e-4,
                                soft_binning_temp=1):

    pass
    # """Computes and returns the soft-binned ECE (binned) tensor.
    # Soft-binned ECE (binned, l2-norm) is defined in equation (11) in this paper:
    # https://arxiv.org/abs/2108.00106. It is a softened version of ECE (binned)
    # which is defined in equation (6).
    # Args:
    #   predictions: tensor of predicted confidences of (batch-size,) shape
    #   labels: tensor of incorrect(0)/correct(1) labels of (batch-size,) shape
    #   soft_binning_bins: number of bins
    #   soft_binning_use_decay: whether temp should be determined by decay factor
    #   soft_binning_decay_factor: approximate decay factor between successive bins
    #   soft_binning_temp: soft binning temperature
    # Returns:
    #   A tensor of () shape containing a single value: the soft-binned ECE.
    # """
    #
    # EPS = 1e-5
    #
    # soft_binning_anchors = torch.tensor(
    #     np.arange(1.0 / (2.0 * soft_binning_bins), 1.0, 1.0 / soft_binning_bins),
    #     dtype=float)
    #
    # predictions = predictions.cpu()
    # predictions_tile = torch.tile(torch.unsqueeze(predictions, 1), (1, soft_binning_anchors.shape[0]))
    # predictions_tile = torch.unsqueeze(predictions_tile, 2)
    # bin_anchors_tile = torch.tile(
    #     torch.unsqueeze(soft_binning_anchors, 0), (predictions.shape[0], 1))
    # bin_anchors_tile = torch.unsqueeze(bin_anchors_tile, 2)
    #
    # if soft_binning_use_decay:
    #     soft_binning_temp = 1 / (
    #             math.log(soft_binning_decay_factor) * soft_binning_bins *
    #             soft_binning_bins)
    #
    # predictions_bin_anchors_product = torch.concat(
    #     [predictions_tile, bin_anchors_tile], axis=2)
    # # pylint: disable=g-long-lambda
    #
    # print(
    #     tf.scan(
    #         fn=lambda _, x: tf.convert_to_tensor([-((x[0] - x[1]) ** 2), 0.]),
    #             elems=tf.ones([15, 2]),
    #             initializer=tf.ones(2)
    #     )
    # )
    #
    # print(tf.scan(fn=lambda _, x: tf.convert_to_tensor([-((x[0] - x[1]) ** 2), 0.]),
    #               elems=tf.ones([15, 2])))
    #
    # # predictions_bin_anchors_differences = torch.sum(
    # #     tf.scan(
    # #         fn=lambda _, row: tf.scan(
    # #             fn=lambda _, x: tf.convert_to_tensor(
    # #                 [-((x[0] - x[1]) ** 2) / soft_binning_temp, 0.]),
    # #             elems=tf.ones([15, 2]),
    # #             initializer=2 * tf.ones(2))
    # #         # initializer=0 * tf.ones(predictions_bin_anchors_product.shape[2:])
    # #     ),
    # #     elems=predictions_bin_anchors_product,
    # #     initializer=tf.zeros(predictions_bin_anchors_product.shape[1:])),
    # #                                       dim = 2,
    # # )
    #
    # # pylint: enable=g-long-lambda
    # predictions_soft_binning_coeffs = F.softmax(
    #     predictions_bin_anchors_differences,
    #     dim=1,
    # )
    #
    # sum_coeffs_for_bin = torch.sum(predictions_soft_binning_coeffs, dim=[0])
    #
    # intermediate_predictions_reshaped_tensor = torch.reshape(
    #     predictions.repeat(soft_binning_anchors.shape),
    #     predictions_soft_binning_coeffs.shape)
    # net_bin_confidence = torch.div(
    #     torch.sum(
    #         intermediate_predictions_reshaped_tensor * predictions_soft_binning_coeffs,
    #         dim=[0]),
    #     torch.max(sum_coeffs_for_bin, EPS * torch.ones(sum_coeffs_for_bin.shape)))
    #
    # intermediate_labels_reshaped_tensor = torch.reshape(
    #     labels.repeat(soft_binning_anchors.shape),
    #     predictions_soft_binning_coeffs.shape)
    # net_bin_accuracy = torch.div(
    #     torch.sum(
    #         intermediate_labels_reshaped_tensor * predictions_soft_binning_coeffs,
    #         dim=[0]),
    #     torch.max(sum_coeffs_for_bin, EPS * tf.ones(sum_coeffs_for_bin.shape)))
    #
    # bin_weights = torch.linalg.norm(sum_coeffs_for_bin, ord=1)[0]
    # soft_binning_ece = torch.sqrt(
    #     torch.tensordot(
    #         torch.square(torch.subtract(net_bin_confidence, net_bin_accuracy)),
    #         bin_weights,
    #         dims=1,
    #     ))
    #
    # return soft_binning_ece
