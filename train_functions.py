import torch
import os
from utils import get_scheduler, get_model, get_lossfn, get_optimizer, get_metric, \
    show_segmentation
from data_functions import get_loaders


def train_epoch(model, train_dl, criterion, metric, optimizer, scheduler, device):
    model.train()
    loss_sum = 0
    score_sum = 0
    for X, y in train_dl:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss = loss.item()
        score = metric(output, y).mean()
        loss_sum += loss
        score_sum += score
    return loss_sum / len(train_dl), score_sum / len(train_dl)


def eval_epoch(model, val_dl, criterion, metric, device):
    model.eval()
    loss_sum = 0
    score_sum = 0
    for X, y in val_dl:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y).item()
            score = metric(output, y).mean()
            loss_sum += loss
            score_sum += score
    return loss_sum / len(val_dl), score_sum / len(val_dl)

'''
def train(model, train_dl, val_dl, criterion, metric, optimizer, scheduler, epochs, save_folder='checkpoints',
          save_name='model', device=None, n_lungs_to_visual=0):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_loss = 999
    best_state_dict = model.state_dict()
    for i in range(epochs):
        print(f'Epoch {i + 1:2}:       Loss       |    IoU metric    ')
        train_loss, train_score = train_epoch(model, train_dl, criterion, metric, optimizer, scheduler, device)
        print(f'Train stats: {train_loss:.6f} | {train_score:.6f}')
        val_loss, val_score = eval_epoch(model, val_dl, criterion, metric, device)
        print(f'  Val stats: {val_loss:.6f} | {val_score:.6f}')
        if n_lungs_to_visual:
            show_segmentation(model, val_dl, n=n_lungs_to_visual)
        if val_loss < last_loss:
            last_loss = val_loss
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, os.path.join(save_folder, save_name) + '.pth')

    model.load_state_dict(best_state_dict)
'''

def run(cfg):
    torch.cuda.empty_cache()

    train_loader, val_loader = get_loaders(cfg)

    print(len(train_loader))
    # нужно сделать
    model = get_model(cfg)
    optimizer = get_optimizer(cfg)
    criterion = get_lossfn(cfg)
    scheduler = get_scheduler(cfg)

    metric = get_metric(cfg)

    trainlosses, vallosses = [], []
    trainmetrics, valmetrics = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(1, cfg.epochs + 1):
        print(f'Epoch #{epoch}')

        train_loss, train_score = train_epoch(model, train_loader, criterion, metric, optimizer, scheduler, device)
        trainlosses.append(train_loss)
        trainmetrics.append(train_score)

        val_loss, val_score = eval_epoch(model, train_loader, criterion, metric, device)
        vallosses.append(val_loss)
        valmetrics.append(val_score)





