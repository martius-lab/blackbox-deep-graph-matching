import torch


def f1_score(tp, fp, fn):
    """
    F1 score (harmonic mix of precision and recall) between predicted permutation matrix and ground truth permutation matrix.
    :param tp: number of true positives
    :param fp: number of false positives
    :param fn: number of false negatives
    :return: F1 score
    """
    device = tp.device

    const = torch.tensor(1e-7, device=device)
    precision = tp / (tp + fp + const)
    recall = tp / (tp + fn + const)
    f1 = 2 * precision * recall / (precision + recall + const)
    return f1


def get_pos_neg(pmat_pred, pmat_gt):
    """
    Calculates number of true positives, false positives and false negatives
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :return: tp, fp, fn
    """
    device = pmat_pred.device
    pmat_gt = pmat_gt.to(device)

    tp = torch.sum(pmat_pred * pmat_gt).float()
    fp = torch.sum(pmat_pred * (1 - pmat_gt)).float()
    fn = torch.sum((1 - pmat_pred) * pmat_gt).float()
    return tp, fp, fn


def get_pos_neg_from_lists(pmat_pred_list, pmat_gt_list):
    device = pmat_pred_list[0].device
    tp = torch.zeros(1, device=device)
    fp = torch.zeros(1, device=device)
    fn = torch.zeros(1, device=device)
    for pmat_pred, pmat_gt in zip(pmat_pred_list, pmat_gt_list):
        _tp, _fp, _fn = get_pos_neg(pmat_pred, pmat_gt)
        tp += _tp
        fp += _fp
        fn += _fn
    return tp, fp, fn


def matching_accuracy_from_lists(pmat_pred_list, pmat_gt_list):
    device = pmat_pred_list[0].device
    match_num = torch.zeros(1, device=device)
    total_num = torch.zeros(1, device=device)
    for pmat_pred, pmat_gt in zip(pmat_pred_list, pmat_gt_list):
        _, _match_num, _total_num = matching_accuracy(pmat_pred, pmat_gt)
        match_num += _match_num
        total_num += _total_num
    return match_num / total_num, match_num, total_num


def matching_accuracy(pmat_pred, pmat_gt):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), "pmat_pred can noly contain 0/1 elements."
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), "pmat_gt should noly contain 0/1 elements."
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num = 0
    total_num = 0

    for b in range(batch_num):
        match_num += torch.sum(pmat_pred[b] * pmat_gt[b])
        total_num += torch.sum(pmat_gt[b])

    return match_num / total_num, match_num, total_num
