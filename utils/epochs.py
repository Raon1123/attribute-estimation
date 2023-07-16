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


def _evaluate_data(model, dataloader, config, device='cpu', masking=False):
    model.eval()

    data_size = len(dataloader.dataset)
    num_classes = dataloader.dataset.num_classes

    method_config = config['METHOD']
    method_name = method_config['name']

    pred = np.zeros((data_size, num_classes))
    masked_targets = np.zeros((data_size, num_classes))
    targets = np.zeros((data_size, num_classes))

    start_idx = 0
    for batch in dataloader:
        data, target = parse_batch(batch, masking=False, device='cpu')
        if masking:
            _, masked_target = parse_batch(batch, masking=masking, device='cpu')
        data = data.to(device)

        with torch.no_grad():
            logits = model.forward(data)
            if method_name in ['LargeLossMatters', 'BoostCAM']:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                preds = torch.sigmoid(logits)

            batch_sz = logits.size(0)
            pred[start_idx:start_idx+batch_sz] = preds.cpu().numpy()
            targets[start_idx:start_idx+batch_sz] = target.numpy()
            if masking:
                masked_targets[start_idx:start_idx+batch_sz] = masked_target.numpy()

            start_idx += batch_sz

    return pred, masked_targets, targets


def evaluate_result(model, 
                    dataloader, 
                    epoch, 
                    config, 
                    device='cpu', 
                    saving=False, 
                    masking=False, 
                    prefix=''):
    """
    Evaluate result of model (mAP, AP etc.)
    """
    pred, masked_targets, targets = _evaluate_data(model, dataloader, config, device, masking)

    # calculate metrics
    ## unmasked
    unmasked_mA = criteria.mean_accuracy(pred, targets)
    unmasked_acc, unmasked_prec, unmasked_recall, unmasked_f1 = criteria.example_based(pred, targets)
    metrics = {
        prefix+'unmasked_mA': unmasked_mA,
        prefix+'unmasked_acc': unmasked_acc,
        prefix+'unmasked_prec': unmasked_prec,
        prefix+'unmasked_recall': unmasked_recall,
        prefix+'unmasked_f1': unmasked_f1,
    }

    ## masked
    if masking:
        masked_mA = criteria.mean_accuracy(pred, masked_targets)
        masked_acc, masked_prec, masked_recall, masked_f1 = criteria.example_based(pred, masked_targets)
        metrics.update({
            prefix+'masked_mA': masked_mA,
            prefix+'masked_acc': masked_acc,
            prefix+'masked_prec': masked_prec,
            prefix+'masked_recall': masked_recall,
            prefix+'masked_f1': masked_f1,
        })
    
    if saving:
        logging.write_metrics(targets, pred, metrics, epoch, config)
        
    return metrics


def evaluate_cam(model, dataloader,
                 num_imgs, 
                 device='cpu'):
    """
    Evaluate CAM of model.
    Input
    - model: model to be evaluated
    - dataloader: dataloader for evaluation

    Output
    - cams: CAMs of model (N, C, H, W)
    """
    model.eval()
    
    cams = []

    cnt_cams = 0
    for batch in dataloader:
        datas, target = parse_batch(batch, device='cpu')
        num_class = target.size(1)
        for data in datas:
            data = data.unsqueeze(0).to(device)

            with torch.no_grad():
                cam = model.get_cam(data)

                img = data.cpu().numpy().squeeze(0)
                cam = cam.cpu().numpy().squeeze(0)
                img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                cam = (cam * 255).astype(np.uint8) 

                # add cam on img
                for single_cam in cam:
                    single_cam = logging.heatmap_on_image(img, single_cam, alpha=0.5)
                    cams.append(single_cam)

            cnt_cams += data.shape[0]
            if cnt_cams >= num_imgs:
                break

    # return as pytorch tensor
    ret = np.concatenate(cams, axis=0)
    return torch.from_numpy(ret)