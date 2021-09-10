import torch
import os
from utils import get_scheduler, get_model, get_criterion, get_optimizer, get_metric
from data_functions import get_loaders
from utils import OneHotEncoder
import wandb


def train_epoch(model, train_dl, encoder, criterion, metric, optimizer, scheduler, device):
    model.train()
    loss_sum = 0
    score_sum = 0
    for X, y in train_dl:
        X = X.to(device)
        if len(torch.unique(X)) == 1:
            continue
        if encoder is not None:
            y = encoder(y)
        y = y.squeeze()
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


def eval_epoch(model, val_dl, encoder, criterion, metric, device):
    model.eval()
    loss_sum = 0
    score_sum = 0
    for X, y in val_dl:
        X = X.to(device)
        if len(torch.unique(X)) == 1:
            continue
        if encoder is not None:
            y = encoder(y)
        y = y.squeeze()
        y = y.to(device)

        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y).item()
            score = metric(output, y).mean()
            loss_sum += loss
            score_sum += score
    return loss_sum / len(val_dl), score_sum / len(val_dl)


def run(cfg, use_wandb=True, max_early_stopping=2):
    torch.cuda.empty_cache()

    train_loader, val_loader = get_loaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = get_model(cfg)(cfg=cfg).to(device)
    if use_wandb:
        wandb.init(project='Covid19_CT_segmentation_' + str(cfg.dataset_name), entity='aiijcteamname', config=cfg,
                   name=cfg.model)
        wandb.watch(model, log_freq=100)

    optimizer = get_optimizer(cfg)(model.parameters(), **cfg.optimizer_params)
    scheduler = get_scheduler(cfg)(optimizer, **cfg.scheduler_params)

    metric = get_metric(cfg)(**cfg.metric_params)
    criterion = get_criterion(cfg)(**cfg.criterion_params)

    if cfg.output_channels == 1:
        encoder = None
    else:
        encoder = OneHotEncoder(cfg)

    best_val_loss = 999
    last_train_loss = 0
    last_val_loss = 999
    early_stopping_flag = 0
    best_state_dict = model.state_dict()
    for epoch in range(1, cfg.epochs + 1):
        print(f'Epoch #{epoch}')

        train_loss, train_score = train_epoch(model, train_loader, encoder,
                                              criterion, metric,
                                              optimizer, scheduler, device)
        train_score = train_score.item()
        print('      Score   |   Loss')
        print(f'Train: {train_score:.6f} | {train_loss:.6f}')

        val_loss, val_score = eval_epoch(model, train_loader, encoder,
                                         criterion, metric, device)
        val_score = val_score.item()
        print(f'Val: {val_score:.6f} | {val_loss:.6f)}')

        metrics = {'train_score': train_score,
                   'train_loss': train_loss,
                   'val_score': val_score,
                   'val_loss': val_loss,
                   'lr': scheduler.get_last_lr()[-1]}
        if use_wandb:
            wandb.log(metrics)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, os.path.join('checkpoints', cfg.model + '_' + cfg.backbone) + '.pth')
        if train_loss < last_train_loss and val_loss > last_val_loss:
            early_stopping_flag += 1
            if early_stopping_flag == max_early_stopping:
                print('<<< EarlyStopping >>>')
                break
        last_train_loss = train_loss
        last_val_loss = val_loss
    model.load_state_dict(best_state_dict)
    if use_wandb:
        wandb.finish()
    return model
