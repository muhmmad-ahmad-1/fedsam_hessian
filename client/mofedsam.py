from .client import Client
import copy
from models.CNN import CNN
from optimizers.ESAM import ESAM
from utils import *
import torch

class mofedsam(Client):
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        super(mofedsam,self).__init__(dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator,args,epochs)

    def init_optimizer(self, optimizer, learning_rate, weight_decay):
        if optimizer == "sgd":
            self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adam":
            self.base_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = ESAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)
    
    def train(self,momentum):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = type(self.model)(self.num_classes)
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train()
        self.model = self.model.to(self.device)
        
        self.model.zero_grad()

        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.reshape(-1).long().to(self.device)

                self.optimizer.zero_grad()
                self.optimizer.step(alpha=self.args.alpha)
                
                self.optimizer.paras = [batch_x, batch_y, self.loss_func, self.model]
                param_list = param_to_vector(self.model)
                deltas_list = momentum.to(self.device)
                loss_correction = (1-self.args.alpha) * torch.sum(param_list * deltas_list) 
                
                loss_correction.backward()

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()  
                
        self.grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}