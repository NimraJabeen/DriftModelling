import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from core import model
from core.model import loss, scheduler
from core import util

from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trains a model to predict the next density map snapshot.',
        formatter_class=helper.ArgParseFormatter)
    
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('dataset', type=str,
                        help='dataset name')
    parser.add_argument('runid', type=str,
                        help='run id of the model')
    
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for model initialisation and ' 
                        'dataset shuffling')
    parser.add_argument('--fieldname', type=str, default='velocity',
                        help='name of the field to use as input')
    parser.add_argument('--channels', type=int, default=5,
                        help='total number of input channels')
    
    parser.add_argument('--residual', action='store_true',
                        help='predict the residual maps')
    parser.add_argument('--threshold', action='store_true',
                        help='compute the loss only on values above 0')
    parser.add_argument('--interp', type=float, default=0,
                        help='given a value of 0-1, the input field is '
                        'interpolated from t and t+1')
    parser.add_argument('--one-day-only', action='store_false',
                        help='do not input the field at t+1')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1,
                        help='weight decay value')
    parser.add_argument('--batchsize', type=int, default=24,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=8,
                        help='number of epochs')

    parser.add_argument('--scp', type=int, default=2,
                        help='scheduler patience')
    parser.add_argument('--sct', type=float, default=0.01,
                        help='scheduler threshold')
    parser.add_argument('--scn', type=float, default=2,
                        help='max number of scheduler updates')
    
    parser.add_argument('--nw', type=int, default=8,
                        help='number of dataloader workers')
    parser.add_argument('--nmp', action='store_false',
                        help='do not use mixed precision GPU operations')
    parser.add_argument('--debug', action='store_true',
                        help='run the script in debug mode')

    args = parser.parse_args()
    return args


def fetch_datasets(loader, field_name, **kwargs):
    train_set = loader.snapshot_dataset(field_name, subset='train', **kwargs)
    val_set = loader.snapshot_dataset(field_name, subset='val', **kwargs)
    
    return train_set, val_set


def create_dataloaders(train_set, val_set, batch_size, **kwargs):
    train_dataloader = DataLoader(
        train_set, batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_set, batch_size, **kwargs)
    
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    args = parse_args()
    paths = helper.PathIndex(args.directory, args.dataset)  
    loader = helper.Loader(paths)
    
    checkpoint_path = paths.model_dir / args.runid
    torch.manual_seed(args.seed)
    
    # fetch datasets
    train_set, val_set = fetch_datasets(
        loader, args.fieldname, input_map=args.residual,
        field_interp=args.interp, next_field=args.one_day_only)
    
    # create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_set, val_set, batch_size=args.batchsize, pin_memory=True,
        drop_last=True, num_workers=args.nw)
    
    ### initialisations ###
    # neural network
    net = model.models.unet(n_channels=args.channels, n_classes=1)

    # optimiser
    optim = torch.optim.AdamW(
        net.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine decay learning rate scheduler with linear warmup - 
    # optimal warmup time decided based on beta_2 of optimiser
    warmup_epochs = (2/(1-optim.param_groups[0]['betas'][1])) / len(
        train_dataloader)
    scheduler_fn = scheduler.CosineAnnealingWarmRestarts(
        optim, args.epochs-warmup_epochs, warmup_epochs=warmup_epochs)

    # define early stopper if LR scheduler is reduce on plateau
    if scheduler_fn.__class__.__name__ == 'ReduceLROnPlateau':
        early_stopper = util.model.EarlyStopper(optim, args.scn)
    else:
        early_stopper = None
    
    # select GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # loss function
    land_mask = ~loader.glazure64_mesh.mask
    loss_fn = loss.MAE(land_mask, batch_mean=True)
    if args.residual:
        loss_fn = loss.ResidualLoss(loss_fn)

    # training
    trainer = model.Trainer(
        device, args.seed, args.epochs, train_dataloader, val_dataloader, net,
        loss_fn, optim, checkpoint_path, scheduler_fn, args.nmp, early_stopper)
    trainer.train(single_iter=args.debug)
