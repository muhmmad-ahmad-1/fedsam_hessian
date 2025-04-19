from utils.utils_dataset import *
from utils.utils_libs import *
from utils.utils_eval import *
from utils.utils_metric import *
from utils.utils import *
from models import CNN, ResNet, ViT_B_32
from client.client import *
import pandas as pd
import tqdm 
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time


class Args:
    def __init__(self,dataset="CIFAR10",n_clients=10,training_rounds=50,learning_rate=5e-3,global_learning_rate=1,batch_size=64,weight_decay=1e-4,grad_aggregator=True,random_selection=False,
                 optimizer="sgd",loss_func="CELoss",rule="Drichlet",rule_arg=0.0,ub_sgm=0.0,aggr_scheme="FedAvg", local_epochs = 3,max_norm = 10,
                 beta = 0, beta1 = 0, beta2 = 0, alpha = 0, rho = 0, mu = 0, lambd =0 , gamma = 0, epsilon = 0, lr_decay = 0, lambda_t = 0.05, maxIter = 5,
                 model_type = "CNN", sparse_optimizer = "SAM_Eigen_Trace", sparse_type = "Interleaved | Switch", sparse_k = 0.75
                 ):
        
        self.dataset = dataset
        self.n_clients = n_clients
        self.tr_rounds = training_rounds
        self.lr = learning_rate
        self.global_lr = global_learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = 42
        self.data = None
        self.opt_name = optimizer
        self.random_selection = random_selection if random_selection else self.n_clients
        self.loss_func = loss_func
        self.rule = "Drichlet"
        self.split_alpha = rule_arg
        self.local_epochs = local_epochs
        self.ub_sgm = ub_sgm
        self.method = aggr_scheme
        self.grad_aggregator = grad_aggregator
        self.max_norm = max_norm
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.rho = rho
        self.mu = mu
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd
        self.lr_decay = lr_decay
        self.maxIter = maxIter
        self.lambda_t = lambda_t
        self.model_type = model_type
        self.sparse_optimizer = sparse_optimizer
        self.sparse_type = sparse_type
        self.sparse_k = sparse_k


class Server:
    def __init__(self,args : Args, random_init :bool = False , aggregation:str = "default"):
        
        self.dataset = args.dataset
        
        # Initialize model based on dataset and model type
        if args.model_type == "CNN":
            if self.dataset == "CIFAR10":
                self.n_cls = 10
                self.global_model = CNN(n_cls=10)
            elif self.dataset == "CIFAR100":
                self.n_cls = 100
                self.global_model = CNN(n_cls=100)
            elif self.dataset == "PACS":
                self.n_cls = 7
                self.global_model = CNN(n_cls=7)
            elif self.dataset == "OfficeHome":
                self.n_cls = 65
                self.global_model = CNN(n_cls=65)
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset}")
        elif args.model_type == "ResNet":
            if self.dataset == "CIFAR10":
                self.n_cls = 10
                self.global_model = ResNet(n_cls=10)
            elif self.dataset == "CIFAR100":
                self.n_cls = 100
                self.global_model = ResNet(n_cls=100)
            elif self.dataset == "PACS":
                self.n_cls = 7
                self.global_model = ResNet(n_cls=7)
            elif self.dataset == "OfficeHome":
                self.n_cls = 65
                self.global_model = ResNet(n_cls=65)
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset}")
        elif args.model_type == "ViT":
            if self.dataset == "CIFAR10":
                self.n_cls = 10
                self.global_model = ViT_B_32(num_classes=10)
            elif self.dataset == "CIFAR100":
                self.n_cls = 100
                self.global_model = ViT_B_32(num_classes=100)
            elif self.dataset == "PACS":
                self.n_cls = 7
                self.global_model = ViT_B_32(num_classes=7)
            elif self.dataset == "OfficeHome":
                self.n_cls = 65
                self.global_model = ViT_B_32(num_classes=65)
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset}")
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
            
        self.n_clients = args.n_clients
        self.tr_rounds = args.tr_rounds
        self.lr = args.lr
        self.global_lr = args.global_lr
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.seed = 42
        self.data = None
        self.opt_name = args.opt_name
        self.random_init = random_init
        self.aggregation = aggregation
        self.random_selection = args.random_selection if args.random_selection else self.n_clients
        self.local_epochs = args.local_epochs
        self.lr_decay = args.lr_decay
        self.max_norm  = args.max_norm
        
        if args.loss_func == "CELoss":
            self.loss_func = torch.nn.CrossEntropyLoss()

        self.rule = args.rule
        if self.rule =="Drichlet":
            self.alpha = args.split_alpha
        
        self.aggregator = args.method
        self.ub_sgm = args.ub_sgm
        self.grad_aggregator = args.grad_aggregator
        self.args = args
        
        self.clients = []
        self.data_clients = []
        
        self.test = None
        self.global_grad = {name:torch.zeros_like(grad) for name,grad in self.global_model.named_parameters()}
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.metrics = {}
        self.correlations = {}

        self.global_model.to(self.device)
        
        self.split_data()
        
    def run_experiment(self):
        
        self.initialize_clients()
        # Only applicable for MoFedSAM
        self.momentum =  param_to_vector(self.global_model)
        
        print("Data Distributed Among Clients")

        self.update_params()
        
        print("Model Parameters Initialized")

        self.train_and_eval()
        self.metrics['args'] = self.args

        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):  # np.float32, np.int32, etc.
                return obj.item()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):  # For class instances
                return make_serializable(vars(obj))
            else:
                return obj
            
        serializable_metrics = make_serializable(self.metrics)
        
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        save_path = "Results/"+self.data.name+"_"+self.aggregation+"_"+"_"+self.aggregator+"_"+self.opt_name+"_"+time+".json"
        with open(save_path, "w") as f:
            json.dump(serializable_metrics, f, indent=4)
        
    def split_data(self):
        self.data = DatasetObject(self.dataset,self.n_clients,self.seed,self.rule,self.ub_sgm,self.alpha)
        path = "Data/"+self.data.name
        self.trn_x = np.load(path+"/clnt_x.npy")
        self.trn_y = np.load(path+"/clnt_y.npy")
        self.test_x = np.load(path+"/tst_x.npy")
        self.test_y = np.load(path+"/tst_y.npy")

        self.data_loader = data.DataLoader(Dataset(self.test_x,self.test_y,dataset_name=self.dataset),self.batch_size,False)
        
        self.data_cardinality = np.array([self.trn_y[i].shape[0] for i in range(self.n_clients)])
        self.data_ratio = self.data_cardinality / np.sum(self.data_cardinality)
        self.test = Test(self.dataset,self.test_x,self.test_y,self.global_model,self.device,self.tr_rounds)
    
    def initialize_clients(self):
         self.clients = [Client(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,10,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def update_params(self):
        for client in self.clients:
            client.update_parameters(self.global_model.parameters()) 
    
    def aggregate_grad(self):
        '''
        Uses gradient based aggregation scheme to compute global gadient
        If gradient based aggregator for parameter calculation, step the gradient to get global parameter
        '''

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
    
    def aggregate_params(self):
        '''
        Uses parameter-based aggregation schemes to get global parameters
        '''
        # if self.aggregation == "smart":
        #     self.smart_aggregation()

        if self.aggregation == "default":
            aggregated_params = [torch.zeros_like(param) for param in self.clients[0].model.parameters()]
            
            for j,client in enumerate(self.clients):
                    for i, grad in enumerate(client.model.parameters()):
                        aggregated_params[i] += grad * self.data_ratio[j]
                        
            with torch.no_grad():
                for param, aggregated_param in zip(self.global_model.parameters(),aggregated_params):
                    param.data.copy_(aggregated_param)
        
    
    # def smart_aggregation(self):
    #     self.global_model = self.global_model.to(self.device)

    #     client_params = [dict(client.model.named_modules()) for client in self.clients]

    #     aggregated_params = aggregate_most_correlated(client_params,self.data_ratio)

    #     # Prepare the state dict to load into the model
    #     state_dict = {}

    #     for name, module in aggregated_params.items():
    #         for param_name, param in module.named_parameters(recurse=False):
    #             # Create the key as 'module_name.param_name'
    #             full_param_name = f"{name}.{param_name}"
    #             state_dict[full_param_name] = torch.nn.Parameter(param.data)

    #     # Load the state dict into the model
    #     self.global_model.load_state_dict(state_dict)
    
    def train(self):
        # for client in self.clients:
        #     client.train()
        self.random_select = random.sample(range(self.n_clients),self.random_selection)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.train) for i,client in enumerate(self.clients) if i in self.random_select]

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Client training failed: {e}")
        
    def test_metrics(self,round):
        acc,f1, trace, top_eigenvalue =  self.test.test(round)
        
        self.metrics[round]["Accuracy"] = acc
        self.metrics[round]["F1 Score"] = f1
        if trace is not None:
            self.metrics[round]["Trace"] = trace
        if top_eigenvalue is not None:
            self.metrics[round]["Top Eigenvalue"] = top_eigenvalue
    
    def gradient_eval(self,round):
        client_grads = [client.grad for client in self.clients if self.clients.index(client) in self.random_select]
        self.grad_metrics = Gradient_Metrics(self.random_selection,client_grads,self.global_grad)
        
        self.metrics[round] = self.grad_metrics.metrics
    
    def train_and_eval(self):
        for r in range(self.tr_rounds):
            start_time = time.time()
            # Global model communication and local training
            print("Training: Round",r+1,"/",self.tr_rounds)
            self.train()
            
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
            self.metrics[r]["Time"] = time.time() - start_time
            print("Training Round Complete!")