import torch
from client import *
from .server import Server,Args
from utils import *
from concurrent.futures import ThreadPoolExecutor

class FedSAM_S(Server):
    def __init__(self,args: Args, random_init :bool = False , aggregation:str = "default"):
        super(FedSAM_S,self).__init__(args,random_init,aggregation)
        self.params = get_mdl_params(self.global_model).to(self.device)
        self.c_i = torch.zeros(self.n_clients,self.params.shape[0]).to(self.device)
        self.c = torch.zeros(self.params.shape[0]).to(self.device)
        self.delta_c = torch.zeros(self.params.shape[0]).to(self.device)
        self.aggregator = "FedSAM_S"
        
    def initialize_clients(self):
         self.clients = [fedsam_s(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def aggregate_grad(self):
        self.global_model = self.global_model.to(self.device)
        
        aggregated_gradients = {name:torch.zeros_like(grad).to(self.device) for name,grad in self.clients[0].grad.items()}
        
        for j,client in enumerate(self.clients):
            if j in self.random_select:
                for i,(name,grad) in enumerate(client.grad.items()):
                    aggregated_gradients[name] += grad * self.data_ratio[j]
        
        with torch.no_grad():
            for param, agg_grad in zip(self.global_model.parameters(), aggregated_gradients.values()):
                param.grad = agg_grad
        
        self.global_grad = {name:grad*self.global_lr for name,grad in aggregated_gradients.items()}
        
        if self.grad_aggregator:
            for name,param in self.global_model.named_parameters():
                param.data.copy_(param-self.global_lr*param.grad)
        
        update = []
        for name,param in aggregated_gradients.items():
            update.append(param.clone().detach().cpu().reshape(-1))
        
        update = torch.cat(update)
        
        for j,client in enumerate(self.clients):
            if j in self.random_select:
                c_i_update = self.c_i[j] - self.c +\
                                dict_to_vector(self.clients[j].grad).clone().detach() /self.local_epochs / self.lr
                self.delta_c += c_i_update - self.c_i[j]
                self.c_i[j] = c_i_update
            
        self.c += self.delta_c / len(self.random_select)
        self.delta_c *= 0
        
        
                 
    def train(self):
        self.random_select = random.sample(range(self.n_clients),self.random_selection)
        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(client.train(self.global_grad,self.c-self.c_i[i])) for i,client in enumerate(self.clients) if i in self.random_select]
        #     for future in futures:
        #         try:
        #             future.result()
        #         except Exception as e:
        #             print(f"Client training failed: {e}")
        for i, client in enumerate(self.clients):
            if i in self.random_select:
                client.train(self.global_grad,self.c-self.c_i[i])