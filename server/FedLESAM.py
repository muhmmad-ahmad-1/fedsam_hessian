import torch
from client import *
from .server import Server,Args
from utils import *
from concurrent.futures import ThreadPoolExecutor

class FedLESAM(Server):
    def __init__(self,args: Args, random_init :bool = False , aggregation:str = "default"):
        super(FedLESAM,self).__init__(args,random_init,aggregation)
        self.aggregator = "FedLESAM"
        
    def initialize_clients(self):
         self.clients = [fedlesam(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
         
    
    def train(self):
        self.random_select = random.sample(range(self.n_clients),self.random_selection)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.train(self.global_grad,self.c-self.c_i[i])) for i,client in enumerate(self.clients) if i in self.random_select]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Client training failed: {e}")