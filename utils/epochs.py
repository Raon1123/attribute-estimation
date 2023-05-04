import numpy as np
import torch
import pickle

import utils.criteria as criteria

def train_epoch(model, train_dataloader, optimizer, config, device='cpu'):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to be trained.
        train_dataloader (DataLoader): Dataloader for training.
        optimizer (Optimizer): Optimizer for training.
        device (str): Device to be used for training.

    Returns:
        float: Average loss of the epoch.
    """
    model.train()
    train_loss = 0.0
    for (data, target) in train_dataloader:
        data, target = data.to(device).float(), target.to(device)

        optimizer.zero_grad()
        if config['METHOD']['name'] == 'LargeLossMatters':
            loss, correction_idx = model.loss(data, target)
            if config['METHOD']['mod_scheme'] == 'LL-Cp' and correction_idx is not None:
                raise NotImplementedError
        else:
            loss = model.loss(data, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_dataloader)


def test_epoch(model, test_dataloader, config, device='cpu'):
    """
    Test the model for one epoch.

    Args:
        model (nn.Module): Model to be tested.
        test_dataloader (DataLoader): Dataloader for testing.
        device (str): Device to be used for testing.

    Returns:
        loss: Average loss of the epoch.
    """
    model.eval()
    test_loss = 0.0
    for (data, target) in test_dataloader:
        data, target = data.to(device).float(), target.to(device)

        with torch.no_grad():
            if config['METHOD']['name'] == 'LargeLossMatters':
                loss, _ = model.loss(data, target)
            else:
                loss = model.loss(data, target)

            test_loss += loss.item()

    return test_loss / len(test_dataloader)


def update_epoch(model, config):
    """
    Update function after each epoch.
    """

    model_config = config['METHOD']

    if model_config['name'] == 'LargeLossMatters':
        model.decrease_clean_rate()


def evaluate_result(model, test_dataloader, epoch, config, writer, device='cpu'):
    """
    Evaluate result of model (mAP, AP etc.)
    """
    model.eval()

    data_size = len(test_dataloader.dataset)
    num_classes = test_dataloader.dataset.get_num_classes()

    pred, gt = np.zeros((data_size, num_classes)), np.zeros((data_size, num_classes))
    start_idx = 0
    for (data, target) in test_dataloader:
        data = data.to(device).float()

        with torch.no_grad():
            logits = model.forward(data)
            if config['METHOD']['name'] == 'LargeLossMatters':
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                preds = torch.sigmoid(logits)

            batch_sz = logits.size(0)
            pred[start_idx:start_idx+batch_sz] = preds.cpu().numpy()
            gt[start_idx:start_idx+batch_sz] = target.numpy()

            start_idx += batch_sz

    mA = criteria.mean_accuracy(pred, gt)
    acc, prec, recall, f1 = criteria.example_based(pred, gt)

    metrics = {'mA': mA, 'acc': acc, 'prec': prec, 'recall': recall, 'f1': f1}

    return metrics