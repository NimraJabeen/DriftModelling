import numpy as np
import pandas as pd
import torch
from pathlib import Path

from core.model.UNet import model as _unet
from core.util import misc


_DP = 11 # decimal print precision
_PRINT_STEP = 100 # number of batches before printing the loss


# =============================================================================
# models
# =============================================================================


class _Models:
    def __getitem__(self, x):
        return getattr(self, x)
    
    class _Model:
        def __init__(self, model, extract_fn, *args, out_modifier=None,
                     **kwargs):
            self.net = model(*args, **kwargs)
            self.__extract = extract_fn
            if out_modifier is not None:
                self.extract = lambda y: out_modifier(self.__extract(y))
            else:
                self.extract = self.__extract
                   
    class unet(_Model):
        def __init__(self, *args, **kwargs):
            super().__init__(
                _unet.UNet, lambda y: y.squeeze(1),
                *args, **kwargs)    
                
models = _Models()


# =============================================================================
# training and evaluation
# =============================================================================


class _Model:
    def __init__(self, device, net, loss_fn, checkpoint_path, mixed_precision):
        self._device = device
        self._net = net.net
        self._extract = net.extract
        self._loss_fn = loss_fn
        self._checkpoint_path = Path(checkpoint_path)
        self._mp = mixed_precision
        self.epoch = 0
        
        self._model_to_device()
    
    @property
    def loss_out_path(self):
        return self._checkpoint_path / 'epoch_loss.txt'
    
    @property
    def best_epoch(self):
        losses = pd.read_csv(self.loss_out_path, index_col='epoch')
        best_idx = losses.val_loss.argmin()
        best_epoch = losses.iloc[best_idx].name
        
        return best_epoch
    
    @staticmethod
    def _path_to_epoch(path):
        return int(path.stem)
    
    def _model_to_device(self):
        self._net.to(self._device)
        print('Model sent to {}'.format(self._device))
    
    def send_to_device(self, X, y, args):
        X = X.to(self._device)
        y = y.to(self._device)
        args = [arg.to(self._device) for arg in args]
        
        return X, y, args
    
    def get_loss(self, X, y, *args):
        # compute prediction and loss
        pred = self.predict(X)
        loss = self._loss_fn(pred, y, *args)

        return loss
    
    def predict(self, X):
        return self._extract(self._net(X))
    
    def get_checkpoint_path(self, epoch):
        return (self._checkpoint_path / str(epoch)).with_suffix('.pt')
    
    def load_checkpoint(self, epoch, weights_only=False):
        path = self.get_checkpoint_path(epoch)
        checkpoint = torch.load(path, map_location=torch.device(self._device))
        self._net.load_state_dict(checkpoint['model_state_dict'])
        
        if not weights_only:
            self.epoch = epoch
            if hasattr(self, '_optim'):
                self._optim.load_state_dict(checkpoint['optim_state_dict'])
            if hasattr(self, '_scheduler') and self._scheduler is not None:
                self._scheduler.load_state_dict(checkpoint['scheduler'])
            if hasattr(self, '_scaler') and 'scaler' in checkpoint:
                self._scaler.load_state_dict(checkpoint['scaler'])
            if hasattr(self, '_early_stopper') and (
                    self._early_stopper is not None):
                self._early_stopper.load_state_dict(
                    checkpoint['early_stopper'])
                
        print("Loaded epoch {} from checkpoint".format(epoch))

    def load_last_checkpoint(self):
        try:
            path = max(self._checkpoint_path.glob('*.pt'),
                       key=self._path_to_epoch)
            epoch = self._path_to_epoch(path)
            self.load_checkpoint(epoch)
        except ValueError:
            pass
        
    def load_best_checkpoint(self, weights_only=False):
        self.load_checkpoint(self.best_epoch, weights_only)


class Trainer(_Model):
    def __init__(self, device, seed, epochs, train_dataloader, val_dataloader,
                 net, loss_fn, optim, checkpoint_path, scheduler=None,
                 mixed_precision=True, early_stopper=None,
                 remove_old_checkpoints=True):
        super().__init__(device, net, loss_fn, checkpoint_path, mixed_precision)
        self._evaluator = Evaluator(device, net, val_dataloader, loss_fn,
                                    checkpoint_path, mixed_precision)
        self._seed = seed
        self._epochs = epochs
        self._dataloader = train_dataloader
        self._optim = optim
        self._scheduler = scheduler
        self._scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self._early_stopper = early_stopper
        self._remove_old_checkpoints = remove_old_checkpoints

        self._set_seed()

    def _set_seed(self):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

    def _train_loop(self, epoch, single_iter, print_step):
        self._net.train()

        iters = len(self._dataloader)
        size = iters * self._dataloader.batch_size

        running_loss = 0
        print_loss = 0
        for i, (X, y, *args) in enumerate(self._dataloader, start=1):
            X, y, args = self.send_to_device(X, y, args)
            # compute loss
            with torch.cuda.amp.autocast(enabled=self._mp):
                loss = self.get_loss(X, y, *args)

            # backprop
            self._optim.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward()
            
            # update params and learning rate
            self._scaler.step(self._optim)
            self._scaler.update()
            if self._scheduler.__class__.__name__ != 'ReduceLROnPlateau':
                self._scheduler.step((epoch-1) + i / iters)

            running_loss += loss.item() * len(X)
            print_loss += loss.item()
            # print average loss since the last print
            if i % print_step == 0:
                n_trained = self._dataloader.batch_size * i
                mean_loss = print_loss / print_step
                print_loss = 0
                
                print('{}/{}; Loss: {:.{}f}'.format(
                    n_trained, size, mean_loss, _DP))
            if single_iter:
                break
        
        return running_loss / size

    def _save_checkpoint(self, epoch, train_loss):
        path = self.get_checkpoint_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self._net.state_dict(),
            'optim_state_dict': self._optim.state_dict(),
            'scaler': self._scaler.state_dict(),
            'train_loss': train_loss
            }
        if self._scheduler is not None:
            checkpoint['scheduler'] = self._scheduler.state_dict()
        if self._early_stopper is not None:
            checkpoint['early_stopper'] = self._early_stopper.state_dict()
            
        torch.save(checkpoint, path)
        print('Checkpoint saved at {}'.format(path))
        
    def _cleanup_checkpoints(self, epoch): 
        for path in self._checkpoint_path.glob('*.pt'):
            if int(path.stem) not in (epoch, self.best_epoch):
                path.unlink()
        
    def _output_loss(self, epoch, train_loss, val_loss):
        with open(self.loss_out_path, 'a') as file:
            if epoch == 1:
                file.write('epoch,train_loss,val_loss\n')
            file.write('{},{},{}\n'.format(epoch, train_loss, val_loss))
        
        s = 'Train loss: {:.{}f}\nVal loss: {:.{}f}'
        print(s.format(train_loss, _DP, val_loss, _DP))

    def _update_scheduler(self, val_loss):
        if self._scheduler is not None:
            scheduler_args = []
            if self._scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler_args.append(val_loss)
            self._scheduler.step(*scheduler_args)
            
    def _update_and_check_early_stopper(self):
        if self._early_stopper is not None:
            updates = self._early_stopper.update()
            if updates is not None:
                if 0 < updates < self._early_stopper.max_updates:
                    self.load_best_checkpoint(weights_only=True)
                elif updates == self._early_stopper.max_updates:
                    return True
    
    def train(self, single_iter=False):
        if single_iter:
            print("/!\ Warning /!\ -- set to single iteration mode\n")
        
        self.load_last_checkpoint()
        starting_epoch = self.epoch + 1
        for epoch in range(starting_epoch, self._epochs+1):
            if self._update_and_check_early_stopper():
                print('Early stopping criteria met')
                break
            
            print('Epoch {}/{} {}'.format(epoch, self._epochs, '-'*30))
            print_step = _PRINT_STEP if not single_iter else 1
            
            train_loss = self._train_loop(epoch, single_iter, print_step)
            val_loss = self._evaluator.evaluate(single_iter)
            
            if self._scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self._update_scheduler(val_loss)
            self._save_checkpoint(epoch, train_loss)
            self._output_loss(epoch, train_loss, val_loss)
            self._cleanup_checkpoints(epoch)
            

            
class Evaluator(_Model):
    def __init__(self, device, net, dataloader, loss_fn, checkpoint_path,
                 mixed_precision):
        super().__init__(device, net, loss_fn, checkpoint_path, mixed_precision)
        self._dataloader = dataloader
        
    def _get_sample_loss_path(self, prefix):
        fname = 'loss.txt'
        if prefix is not None:
            fname = '{}_'.format(prefix) + fname
        return self._checkpoint_path / fname
    
    def _write_sample_losses(self, indices, losses, loss_names,
                             write_header, prefix):
        path = self._get_sample_loss_path(prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as file:
            if write_header:
                file.write('index,{}\n'.format(','.join(loss_names)))
            for index, line in zip(indices, zip(*losses)):
                line = ','.join(map(lambda x: str(x.item()), line))
                file.write('{},{}\n'.format(index, line))
                
    def evaluate(self, single_iter):
        self._net.eval()

        loss = 0
        with torch.no_grad():
            for X, y, *args in self._dataloader:
                X, y, args = self.send_to_device(X, y, args)
                # compute loss for batch
                with torch.cuda.amp.autocast(enabled=self._mp):
                    loss += self.get_loss(X, y, *args).item() * len(X)
                if single_iter:
                    break
                
        # compute loss for dataset
        size = len(self._dataloader.dataset)
        if self._dataloader.drop_last:
            size -= size % self._dataloader.batch_size
        loss /= size
                
        return loss
    
    def save_results(self, loss_names='loss', prefix=None, loss_fns=None,
                     residual=False, clip=True):
        self._net.eval()

        if loss_fns is None:
            loss_fns = [self._loss_fn]
        if type(loss_fns) is not list:
            loss_fns = [loss_fns]
        if type(loss_names) is not list:
            loss_names = [loss_names]
        
        indices = iter(self._dataloader.dataset.indices)
        size = len(self._dataloader.dataset)
        with torch.no_grad():
            pred = None
            for i, (X, y, *args) in enumerate(self._dataloader, start=1):
                X, y, args = self.send_to_device(X, y, args)
                
                with torch.cuda.amp.autocast(enabled=self._mp):
                    pred = self.predict(X)

                if residual:
                    pred += X[:, -1]
                if clip:
                    pred = pred.clamp(0)
            
                losses = []
                for loss_fn in loss_fns:
                    losses.append(loss_fn(pred, y, *args))
                    
                batch_indices = misc.yield_n(indices, len(X))
                self._write_sample_losses(
                    batch_indices, losses, loss_names, i==1, prefix)

                if i % _PRINT_STEP == 0:
                    n = self._dataloader.batch_size * i
                    print('{}: {}/{}'.format(loss_names, n, size))
    
