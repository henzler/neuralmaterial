import torch
from torch import nn

class CoreModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def print_num_params(self):

        num_param_dict = {}
        for k in self.module_list:
            params = list(getattr(self,k).parameters())
           
            if len(params) > 0:                
                num_param_dict[k] = torch.stack([
                    *[(torch.prod(torch.tensor(p.size()))) for p in params]
                    ], 0).sum().item()
            else:
                num_param_dict[k] = 0
        
        print('-'*10)
        print('[Model]')
        for k, v in num_param_dict.items(): print(f'   {k} | params: {v:,}')
        print(f'   total params: {sum(num_param_dict.values()):,}')
        print('-'*10)

    def _register_module_list(self):
        self.module_list = [
            attr for attr in dir(self)
                if isinstance(getattr(self, attr), nn.Module)
        ]

    def register_device(self, device):
        [
            getattr(self, module_name).to(device) 
                for module_name in self.module_list
        ]

    def _cache_for_logger(self, inputs, outputs):
        self.inputs_cache = inputs
        self.outputs_cache = {k: v.detach().cpu() for k, v in outputs.items()}

    def training_start(self):
        self._register_module_list()
        self.global_step = 0
        self.optimizer = self.configure_optimizer()
    
    def finetuning_start(self):
        self._register_module_list()
        self.global_step = 0
        self.optimizer = self.configure_optimizer_finetuning()

    def after_train_step(self):
        self.global_step += 1

    def training_end(self):
        print('[INFO] training complete.')
    
    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

