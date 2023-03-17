import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn

from core import model
from core.model import loss
from core import util

from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluates a model that predicts density map snapshots.',
        formatter_class=helper.ArgParseFormatter)
    
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('dataset', type=str, nargs='+',
                        help='dataset name(s)')
    parser.add_argument('runid', type=str, default=0,
                        help='run id of the model')
    parser.add_argument('--subset', type=str, default='test',
                        help='evaluation subset')
    
    parser.add_argument('--fieldname', type=str, default='velocity',
                        help='name of the field to use as input')
    parser.add_argument('--channels', type=int, default=5,
                        help='total number of input channels')
    
    parser.add_argument('--one-day-only', action='store_false',
                        help='do not input the field at t+1')
    parser.add_argument('--interp', type=float, default=0,
                        help='given a value of 0-1, the input field is '
                        'interpolated from t and t+1')
    
    parser.add_argument('--residual', action='store_true',
                        help='evaluate residual maps')
    parser.add_argument('--threshold', action='store_true',
                        help='compute the loss only on values above 0')
    
    parser.add_argument('--batchsize', type=int, default=24,
                        help='batch size')
    
    parser.add_argument('--nw', type=int, default=8,
                        help='number of dataloader workers')
    parser.add_argument('--nmp', action='store_false',
                        help='do not use mixed precision GPU operations')
    parser.add_argument('--debug', action='store_true',
                        help='run the script in debug mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    for ds in args.dataset:
        paths = helper.PathIndex(args.directory, ds)  
        loader = helper.Loader(paths)
        checkpoint_path = paths.model_dir / args.runid
        
        # fetch evaluation set and create dataloader
        dataset = loader.snapshot_dataset(
            args.fieldname, subset=args.subset, input_map=args.residual,
            field_interp=args.interp, next_field=args.one_day_only)
        dataloader = DataLoader(
            dataset, args.batchsize, pin_memory=True, num_workers=args.nw)
        
        # select GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialisations
        out_modifier = util.model.MaskedOutput(loader.glazure64_mesh.mask)
        net = model.models.unet(n_channels=args.channels, n_classes=1,
                                out_modifier=out_modifier)
        
        # loss function
        land_mask = ~loader.glazure64_mesh.mask
        loss_fn = loss.MSE(land_mask, batch_mean=False)
    
        # evaluate
        evaluator = model.Evaluator(
            device, net, dataloader, loss_fn, checkpoint_path, args.nmp)
        evaluator.load_best_checkpoint()
        prefix = '{}_{}'.format(ds, args.subset)
        evaluator.save_results(
            prefix=prefix, residual=args.residual, clip=True)

    
    
