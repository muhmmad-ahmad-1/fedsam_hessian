from .client import Client
import copy
from models import *
from optimizers import *
from utils import *

class fedsam_hessian_sparse(Client):
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        super(fedsam_hessian_sparse,self).__init__(dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator,args,epochs)
        self.sparse_type = args.sparse_type
        self.sparse_k = args.sparse_k
    
    def init_optimizer(self, optimizer, learning_rate, weight_decay):
        self.base_optimizer = torch.optim.SGD(self.model.parameters(),learning_rate,weight_decay=weight_decay)
        self.optimizer = self.args.sparse_optimizer
        if self.optimizer == "SAM_Eigen_Trace":
            self.optimizer = SAM_Eigen_Trace(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True,maxIter=self.args.maxIter,lambda_t=self.args.lambda_t)
        elif self.optimizer == "SAM_Hessian":
            self.optimizer = SAM_Hessian(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True,maxIter=self.args.maxIter)
        elif self.optimizer == "SAM_Hessian_Orth":
            self.optimizer = SAM_Hessian_Orth(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True,maxIter=self.args.maxIter)
        elif self.optimizer == "SAM_Trace":
            self.optimizer = SAM_Trace(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True,maxIter=self.args.maxIter,lambda_t=self.args.lambda_t)
        self.fo_optimizer = ESAM(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True)
    
    
    def train(self,round):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = type(self.model)(self.num_classes)
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train(); self.model = self.model.to(self.device)
        
        self.model.zero_grad()

        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if self.sparse_type == "Interleaved" and round % self.sparse_k == 0:
                    optimizer = self.fo_optimizer
                elif self.sparse_type == "Sequential" and round < self.sparse_k * self.args.tr_rounds:
                    optimizer = self.fo_optimizer
                else:
                    optimizer = self.optimizer
                optimizer.zero_grad()
                optimizer.paras = [batch_x,batch_y,self.loss_func,self.model]
                
                optimizer.step()
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),max_norm=self.max_norm)

                self.base_optimizer.step()   
                
        self.grad = { n:(g-l).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}