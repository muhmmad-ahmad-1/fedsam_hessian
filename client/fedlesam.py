from .client import Client
import torch
from optimizers import *
import copy
from models.CNN import CNN

class fedlesam(Client):
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        super(fedlesam,self).__init__(dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator,args,epochs)
    
    def init_optimizer(self, optimizer, learning_rate, weight_decay):
        self.base_optimizer = torch.optim.SGD(self.model.parameters(),learning_rate,weight_decay=weight_decay)
        self.optimizer = LESAM(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True)
    
    def train(self,global_grad):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = type(self.model)(self.num_classes)
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train(); self.model = self.model.to(self.device)
        
        self.model.zero_grad()

        global_update = []
        for param in global_grad.values():
            global_update.append(param.clone().detach().cpu().reshape(-1))
        global_update = torch.cat(global_update)
        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                self.optimizer.paras = [batch_x,batch_y,self.loss_func,self.model]
                
                self.optimizer.step(global_update)
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),max_norm=self.max_norm)

                self.base_optimizer.step()   
                
        self.grad = { n:(g-l).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}