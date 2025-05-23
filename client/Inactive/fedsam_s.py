import torch
from .client import Client
from optimizers import *
from utils import *
import copy
from models.CNN import CNN

def loss_gamma(predictions,labels,param_list,deltas_list,lamb):
    return torch.nn.functional.cross_entropy(predictions,labels,reduction='mean')+torch.sum(param_list * deltas_list)*lamb

class fedsam_s(Client):
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        super(fedsam_s,self).__init__(dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator,args,epochs)
        self.deltas = None
    def init_optimizer(self, optimizer, learning_rate, weight_decay):
        self.base_optimizer = torch.optim.SGD(self.model.parameters(),learning_rate,weight_decay=weight_decay)
        self.optimizer = ESAM_S(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=False)
        self.loss_func = loss_gamma

    def train(self,global_grad,deltas):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = CNN()
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train(); self.model = self.model.to(self.device)
        
        self.model.zero_grad()

        # global_update = []
        # for param in global_grad.values():
        #     global_update.append(param.clone().detach().cpu().reshape(-1))
        # global_update = torch.cat(global_update)
        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                deltas = deltas.to(self.device)
                self.optimizer.zero_grad()
                self.optimizer.paras = [batch_x,batch_y,self.loss_func,self.model,deltas,self.args.lambd]
                
                self.optimizer.step()
                
                torch.nn.utils.clip_grad_norm(parameters=self.model.parameters(),max_norm=self.max_norm)

                self.base_optimizer.step()   
                
        self.grad = { n:(g-l).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}
