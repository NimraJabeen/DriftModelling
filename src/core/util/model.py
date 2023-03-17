class EarlyStopper:
    def __init__(self, optim, max_updates):
        self._optim = optim
        self._lr = self._optim.param_groups[0]['lr']
        self._lr_updates = 0
        
        self.max_updates = max_updates
        
    def update(self):
        lr = self._optim.param_groups[0]['lr']
        if lr != self._lr:
            self._lr = lr
            self._lr_updates += 1
            return self._lr_updates
        
    def state_dict(self):
        state_dict = {'lr': self._lr, 'lr_updates': self._lr_updates,
                      'max_updates': self.max_updates}
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        self._lr = state_dict['lr']
        self._lr_updates = state_dict['lr_updates']
        self.max_updates = state_dict['max_updates']
        
        
class MaskedOutput:
    def __init__(self, mask):
        self._mask = mask
        
    def __call__(self, x):
        x[:, ~self._mask] = 0
        return x