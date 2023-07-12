import numpy as np
import torch
import pickle

import utils.criteria as criteria
import utils.logging as logging

def parse_batch(batch, masking=True, device='cpu'):
    """
    Parse the batch.

    Args:
        batch (tuple): Batch of data and target.
        device (str): Device to be used.

    Returns:
        tuple: Parsed batch.
    """
    data, target, mask = batch
    
    # masking target
    if masking:
        target = target * (1 - mask)

    data, target = data.to(device).float(), target.to(device)
    return data, target


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

    method_config = config['METHOD']
    method_name = method_config['name']

    for batch in train_dataloader:
        data, target = parse_batch(batch, device=device)
        _, un_msk = parse_batch(batch, masking=False, device=device)

        optimizer.zero_grad()
        if method_name in ['LargeLossMatters', 'BoostCAM']:
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

    method_config = config['METHOD']
    method_name = method_config['name']
 
    for batch in test_dataloader:
        data, target = parse_batch(batch, device=device)

        with torch.no_grad():
            if method_name in ['LargeLossMatters', 'BoostCAM']:
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
    method_name = model_config['name']

    if method_name in ['LargeLossMatters', 'BoostCAM']:
        model.decrease_clean_rate()


def evaluate_result(model, test_dataloader, epoch, config, device='cpu', saving=False, masking=False, prefix=''):
    """
    Evaluate result of model (mAP, AP etc.)
    Evaluation without masking
    """
    model.eval()

    data_size = len(test_dataloader.dataset)
    num_classes = test_dataloader.dataset.num_classes

    method_config = config['METHOD']
    method_name = method_config['name']

    pred, gt = np.zeros((data_size, num_classes)), np.zeros((data_size, num_classes))
    start_idx = 0
    for batch in test_dataloader:
        data, target = parse_batch(batch, masking=masking, device='cpu')
        data = data.to(device)

        with torch.no_grad():
            logits = model.forward(data)
            if method_name in ['LargeLossMatters', 'BoostCAM']:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                preds = torch.sigmoid(logits)

            batch_sz = logits.size(0)
            pred[start_idx:start_idx+batch_sz] = preds.cpu().numpy()
            gt[start_idx:start_idx+batch_sz] = target.numpy()

            start_idx += batch_sz

    mA = criteria.mean_accuracy(pred, gt)
    acc, prec, recall, f1 = criteria.example_based(pred, gt)

    if masking:
        prefix = 'masked_'
    elif prefix == '':
        prefix = 'unmasked_'

    metrics = {prefix+'mA': mA, 
               prefix+'acc': acc, 
               prefix+'prec': prec, 
               prefix+'recall': recall, 
               prefix+'f1': f1}

    if saving:
        logging.write_metrics(gt, pred, metrics, epoch, config)

        print("Experiment Result")
        print(f"mA: {mA.mean():.4f}, acc: {acc.mean():.4f}, prec: {prec.mean():.4f}, recall: {recall.mean():.4f}, f1: {f1.mean():.4f}")
        
    return metrics