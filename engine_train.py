# pyre-ignore-all-errors
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
import math
import sys
import torch
import util.misc as misc
from typing import Iterable
import numpy as np
from util.abnormal_utils import filt
import sklearn.metrics as metrics


import typing

def adjust_learning_rate(optimizer, step, len_loader, args: typing.Any):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < args.warmup_epochs * len_loader:
        lr = args.lr * step / max(1, (args.warmup_epochs * len_loader))
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (step - args.warmup_epochs * len_loader) / max(1, ((args.epochs - args.warmup_epochs) * len_loader))))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None, args: typing.Any = None):
    model.train(True)
    model = model.float()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    if epoch >= args.start_TS_epoch:
        model.train_TS = True  # type: ignore
        model.freeze_backbone()  # type: ignore

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, grad_mask, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        grad_mask = grad_mask.to(device, non_blocking=True)

        loss, _, _ = model(samples, grad_mask=grad_mask, targets=targets, mask_ratio=args.mask_ratio)  # type: ignore
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # adjust learning rate
        step = data_iter_step + epoch * len(data_loader)
        lr = adjust_learning_rate(optimizer, step, len(data_loader), args)

        optimizer.zero_grad()
        loss.backward()
        if hasattr(args, 'clip_grad') and args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                   device: torch.device, epoch: int,
                   log_writer=None, args: typing.Any = None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions = []
    labels = []
    videos = []
    import gc
    for i, (samples, samples_abnormal, labels, video_name, _, grads, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if i % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        samples = samples.to(device)
        grads = grads.to(device)
        targets = targets.to(device)
        _, _, _, recon_error = model(samples, grad_mask=grads,targets=targets, mask_ratio=args.mask_ratio)  # type: ignore
        if isinstance(recon_error, list):
            recon_error = recon_error[0] + recon_error[1]
        recon_error = recon_error.detach().cpu().numpy().flatten()
        predictions.extend(recon_error.tolist())

    # Compute statistics
    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = np.nan_to_num(pred, nan=0.)
        if args.dataset=='avenue':
            pred = filt(pred, range=38, mu=11)
        else:
            raise ValueError('Unknown parameters for predictions postprocessing')
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        filtered_preds.append(pred)  # type: ignore
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)  # type: ignore
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)  # type: ignore

    macro_auc = np.nanmean(aucs)

    # Micro-AUC
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    # gather the stats from all processes
    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")
    return {"micro": micro_auc, "macro": macro_auc}
