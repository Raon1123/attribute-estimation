import torch

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
        float: Average loss of the epoch.
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