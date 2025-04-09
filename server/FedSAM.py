import torch
from client import *
from .server import Server,Args

class FedSAM(Server):
    def __init__(self,args: Args, random_init :bool = False , aggregation:str = "default"):
        super(FedSAM,self).__init__(args,random_init,aggregation)
        self.aggregator = "FedSAM"
    
    def initialize_clients(self):
         self.clients = [fedsam(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]