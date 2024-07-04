import os
import numpy as np
import torch
import random
import pandas as pd
from sklearn import metrics
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def init_logger(log_file='train.log'):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger



def padded_cmap(solution, submission, padding_factor=5):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat(
        [solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat(
        [submission, new_rows]).reset_index(drop=True).copy()
    score = metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def map_score(solution, submission):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')
    score = metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',  # 'macro'
    )
    return score



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train_fn(data_loader, model, criterion, optimizer, scheduler, epoch, device, taa_augmentation, cfg=None):

    model.train()
    losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler(enabled=cfg.apex)
    iters = len(data_loader)
    gt = []
    preds = []

    with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
        for i, (data) in t:
            inputs = batch_to_device(data, device)
            targets = data['primary_targets'].to(device)

            inputs['wave'] = taa_augmentation(inputs['wave'].unsqueeze(1))
            inputs['wave'] = inputs['wave'].squeeze(1)

            with autocast(enabled=cfg.apex):
                outputs = model(inputs)
                loss = criterion(outputs['logit'], outputs['target'])

            losses.update(loss.item(), inputs['wave'].size(0))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step(epoch + i / iters)
            t.set_postfix(
                loss=losses.avg,
                grad=grad_norm.item(),
                lr=optimizer.param_groups[0]["lr"]
                )

            gt.append(targets.cpu().detach().numpy())
            preds.append(outputs["logit"].sigmoid().cpu().detach().numpy())

        val_df = pd.DataFrame(
            np.concatenate(gt), columns=cfg.target_columns[:cfg.num_classes])
        pred_df = pd.DataFrame(
            np.concatenate(preds), columns=cfg.target_columns[:cfg.num_classes])
        cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
        cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
        mAP = map_score(val_df, pred_df)

    return losses.avg, cmAP_1, cmAP_5, mAP



def valid_fn(data_loader, model, criterion, epoch, device, cfg=None):
    model.eval()
    losses = AverageMeter()
    gt = []
    preds = []

    with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
        for i, (data) in t:
            inputs = batch_to_device(data, device)
            targets = data['primary_targets'].to(device)

            with autocast(enabled=cfg.apex):
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs['logit'], outputs['target'])

            losses.update(loss.item(), inputs['wave'].size(0))
            t.set_postfix(loss=losses.avg)

            gt.append(targets.cpu().detach().numpy())
            preds.append(outputs["logit"].sigmoid().cpu().detach().numpy())

    val_df = pd.DataFrame(np.concatenate(gt), columns=cfg.target_columns[:cfg.num_classes])
    pred_df = pd.DataFrame(np.concatenate(preds), columns=cfg.target_columns[:cfg.num_classes])
    cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
    cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
    mAP = map_score(val_df, pred_df)

    return losses.avg, cmAP_1, cmAP_5, mAP



def train_loop(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    epochs=10,
    fold=None,
    device=None,
    taa_augmentation=[],
    cfg=None,
    logger=None,
):

    best_score = 0.0
    patience = cfg.early_stopping
    n_patience = 0

    for epoch in range(epochs):

        # train for one epoch
        train_loss, train_score, train_cmAP5, train_mAP = train_fn(
            train_loader, model, criterion, optimizer, scheduler, epoch, device, taa_augmentation, cfg=cfg)

        # evaluate on validation set
        val_loss, val_score, val_cmAP5, val_mAP = valid_fn(
            val_loader, model, criterion, epoch, device, cfg=cfg)

        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Train cmAP1: {train_score:.4f}, Train cmAP5: {train_cmAP5:.4f}, Train mAP: {train_mAP:.4f}, Valid loss: {val_loss:.4f}, Valid cmAP1: {val_score:.4f}, Valid cmAP5: {val_cmAP5:.4f}, Valid mAP: {val_mAP:.4f}")

        is_better = val_score > best_score
        best_score = max(val_score, best_score)

        # Save the best model
        if is_better:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_score,
                "optimizer": optimizer.state_dict(),
            }
            logger.info(
                f"Epoch {epoch} - Save Best Score: {best_score:.4f} Model\n")
            torch.save(
                state,
                os.path.join(cfg.model_output_path, f"fold_{fold}_model.pt")
                )
            n_patience = 0
        else:
            n_patience += 1
            logger.info(
                f"Valid loss didn't improve last {n_patience} epochs.\n")

        if n_patience >= patience:
            logger.info(
                "Early stop, Training End.\n")
            break

    return
