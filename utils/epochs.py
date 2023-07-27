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
        data, target: Parsed data and target.
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
        avg_loss: Average loss of the train epoch.
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
        avg_loss: Average loss of the test epoch.
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

    For instance, LargeLossMatters and BoostCAM update decrease_clean_rate.
    """

    model_config = config['METHOD']
    method_name = model_config['name']

    if method_name in ['LargeLossMatters', 'BoostCAM']:
        model.decrease_clean_rate()


def _evaluate_data(model, dataloader, config, device='cpu', masking=False):
    """
    internal function for evaluate_result
    """
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
    - img_list: images of input (N, C, H, W)
    - cam_list: CAMs of model (N, num_class, H, W)
    """
    model.eval()
    
    img_list = []
    cam_list = []

    target_list = []
    pred_list = []
    mask_list = []

    cnt_cams = 0
    for batch in dataloader:
        datas, targets, masks = batch
        datas = datas.to(device)

        with torch.no_grad():
            batch_cam = model.get_cam(datas)
            preds = model.prediction(datas)
        batch_cam = batch_cam.cpu() # (N, C, 7, 7) note that normalized with [0,1]

        img_list.append(datas.cpu())
        cam_list.append(batch_cam)
        target_list.append(targets.cpu())
        pred_list.append(preds.detach().cpu())
        mask_list.append(masks.cpu())

        cnt_cams += datas.shape[0]
        if cnt_cams >= num_imgs:
            break

    img_list = torch.cat(img_list, dim=0)
    cam_list = torch.cat(cam_list, dim=0)
    cam_list = cam_list * 255.0

    target_list = torch.cat(target_list, dim=0)
    pred_list = torch.cat(pred_list, dim=0)
    mask_list = torch.cat(mask_list, dim=0)

    # return as pytorch tensor
    return img_list, cam_list, target_list, pred_list, mask_list