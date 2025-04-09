import torch
from client import *
from .server import Server
from utils.utils import *
import random

class MoFedSAM(Server):
    def __init__(self,dataset="CIFAR10",n_clients=10,training_rounds=50,learning_rate=1e-1,global_learning_rate = 0.5,batch_size=100,weight_decay=1e-3,optimizer="adam",loss_func="CELoss",rule="Drichlet",rule_arg=0.0,ub_sgm=0.0,aggr_scheme="FedAvg",aggr_args={},random_init = False,aggregation="default",grad_aggregator=True,lr_decay=0):
        super(MoFedSAM,self).__init__(dataset="CIFAR10",n_clients=10,training_rounds=50,learning_rate=1e-1,global_learning_rate = 0.5,batch_size=100,weight_decay=1e-3,optimizer="adam",loss_func="CELoss",rule="Drichlet",rule_arg=0.0,ub_sgm=0.0,aggr_scheme="FedAvg",aggr_args={},random_init = False,aggregation="default",grad_aggregator=True,lr_decay=0)
        self.local_iteration = self.local_epochs * self.data_cardinality[0]/batch_size
    def initialize_clients(self):
         self.clients = [mofedsam(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.opt_name,self.lr,self.weight_decay,10,self.device,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def aggregate_grad(self):
        self.global_model = self.global_model.to(self.device)
        
        aggregated_gradients = {name:torch.zeros_like(grad) for name,grad in self.clients[0].grad.items()}
        
        for j,client in enumerate(self.clients):
            for i,(name,grad) in enumerate(client.grad.items()):
                aggregated_gradients[name] += grad * self.data_ratio[j]
        
        with torch.no_grad():
            for param, agg_grad in zip(self.global_model.parameters(), aggregated_gradients.values()):
                param.grad = agg_grad
        
        self.global_grad = aggregated_gradients
        
        if self.grad_aggregator:
            for name,param in self.global_model.named_parameters():
                param.data.copy_(param-self.global_lr*param.grad)
        
        update = []
        for name,param in aggregated_gradients.items():
            update.append(param.clone().detach().cpu().reshape(-1))
        
        update = torch.cat(update)
        
        self.momentum = update / self.local_iteration / self.lr / self.lr_decay * -1.
    
    def train(self):
        random_select = random.sample(range(self.n_clients),self.random_selection)
        for i,client in enumerate(self.clients):
            if  i in random_select:
                client.train(self.momentum)
        
        