import torch
from client import *
from .server import *

class FedAdam(Server):
    def __init__(self,args : Args, random_init :bool = False , aggregation:str = "default"):
        super(FedAdam,self).__init__(args,random_init,aggregation)
        self.m = {name:torch.zeros_like(param) for name,param in self.global_model.named_parameters() if param.requires_grad}
        self.v = {name:torch.zeros_like(param) for name,param in self.global_model.named_parameters() if param.requires_grad}
    
    def initialize_clients(self):
         self.clients = [fedadam(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    def aggregate_grad(self):
        self.global_model = self.global_model.to(self.device)
        
        aggregated_gradients = {name:torch.zeros_like(grad) for name,grad in self.clients[0].grad.items()}
        
        for j,client in enumerate(self.clients):
            for i,(name,grad) in enumerate(client.grad.items()):
                aggregated_gradients[name] += grad * self.data_ratio[j]
        
        self.m = {name: self.args.beta1* params +(1-self.args.beta1)*aggregated_gradients[name] for name, params in self.m.items()}
        self.v = {name: self.args.beta1* params +(1-self.args.beta1)*(aggregated_gradients[name])**2 for name, params in self.v.items()}
        
        with torch.no_grad():
            for name,param in self.global_model.named_parameters():
                param.grad = self.m[name] / (self.v[name].sqrt()+self.args.epsilon)
        
        self.global_grad = {name:param.grad for name,param in self.global_model.named_parameters()}
        
        if self.grad_aggregator:
            for name,param in self.global_model.named_parameters():
                param.data.copy_(param+self.global_lr*param.grad)
                
    def aggregate_params(self):
        return super().aggregate_params()
        