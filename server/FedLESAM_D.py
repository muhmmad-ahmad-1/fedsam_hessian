import torch
from client import *
from .server import Server,Args
from utils import *
from concurrent.futures import ThreadPoolExecutor

class FedLESAM_D(Server):
    def __init__(self,args: Args, random_init :bool = False , aggregation:str = "default"):
        super(FedLESAM_D,self).__init__(args,random_init,aggregation)
        self.params = get_mdl_params(self.global_model).to(self.device)
        self.h = torch.zeros(self.n_clients,self.params.shape[0]).to(self.device)
        self.aggregator = "FedLESAM_D"
        
    def initialize_clients(self):
         self.clients = [fedlesam_d(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def aggregate_grad(self):
        self.global_model = self.global_model.to(self.device)
        
        aggregated_gradients = {name:torch.zeros_like(grad).to(self.device) for name,grad in self.clients[0].grad.items()}
        
        
        for j,client in enumerate(self.clients):
            if j in self.random_select:
                self.h[j] += dict_to_vector(client.grad).clone().detach()
                for i,(name,grad) in enumerate(client.grad.items()):
                    aggregated_gradients[name] += grad * self.data_ratio[j]
        
        update = []
        for name,param in aggregated_gradients.items():
            update.append(param.clone().detach().reshape(-1))
        
        update = torch.cat(update) + torch.mean(self.h, dim=0)
        
        updated_dict = vector_to_dict(aggregated_gradients,update)

        self.global_grad = {name:grad*self.global_lr for name,grad in updated_dict.items()}
        
        with torch.no_grad():
            for param, agg_grad in zip(self.global_model.parameters(), updated_dict.values()):
                param.grad = agg_grad
        
        if self.grad_aggregator:
            for name,param in self.global_model.named_parameters():
                param.data.copy_(param-self.global_lr*param.grad)
        
                 
    def train(self):
        self.random_select = random.sample(range(self.n_clients),self.random_selection)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.train(self.global_grad,self.h[i]-get_mdl_params(client.model).clone().detach())) for i,client in enumerate(self.clients) if i in self.random_select]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Client training failed: {e}")
        # for i,client in enumerate(self.clients):
        #     if i in self.random_select:
        #         client.train(self.global_grad,self.h[i]-get_mdl_params(client.model).clone().detach())