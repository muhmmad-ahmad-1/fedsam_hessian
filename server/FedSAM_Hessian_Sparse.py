import torch
from client import *
from .server import Server,Args
import random
from concurrent.futures import ThreadPoolExecutor

class FedSAM_Hessian_Sparse(Server):
    def __init__(self,args: Args, random_init :bool = False , aggregation:str = "default"):
        super(FedSAM_Hessian_Sparse,self).__init__(args,random_init,aggregation)
        self.aggregator = "FedSAM_Hessian_Sparse"
    
    def initialize_clients(self):
         self.clients = [fedsam_hessian_sparse(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def train(self,round):
        self.random_select = random.sample(range(self.n_clients),self.random_selection)
        
        # Use ThreadPoolExecutor for parallel client training
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.train, round) for i, client in enumerate(self.clients) if i in self.random_select]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Client training failed: {e}")
    
    def train_and_eval(self):
        for r in range(self.tr_rounds):
            # Global model communication and local training
            print("Training: Round",r+1,"/",self.tr_rounds)
            self.train(r)
            
            print("Local Training Completed")
            #Global Gradient calculation (and update if relevant scheme) and related metric calculations
            self.aggregate_grad()
            self.gradient_eval(r)
            
            #Parametric aggregation (if not gradient based aggregation)
            if not self.grad_aggregator:
                self.aggregate_params()
            print("Global Aggregation Completed")

            #param_metrics = Parameter_Metrics(self.n_clients,[dict(client.model.named_parameters()) for client in self.clients],dict(self.global_model.named_parameters()))
            #self.metrics[r].update(param_metrics.metrics)
            # Send model update for next round
            self.update_params()
            
            #Record performance on test data (F1 and Accuracy)
            self.test_metrics(r)
            print("Training Round Complete!")