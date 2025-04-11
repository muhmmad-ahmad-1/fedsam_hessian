from models import *
from utils.utils_dataset import *
from utils.utils_libs import *
import torch
from optimizers import SAM, ESAM, GAMASAM, LESAM
import copy
from torch.utils import data

class Client:
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        # Initialize common attributes
        self.trn_x = trn_x
        self.trn_y = trn_y
        self.loss_func = loss_func
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_norm = max_norm
        self.args = args
        self.deltas = None
        self.dataset = Dataset(self.trn_x,self.trn_y,True,dataset)
        self.dataloader = data.DataLoader(self.dataset,batch_size=batch_size,shuffle=True)
        self.grad_aggregation = grad_aggregator
        self.epochs = epochs
        self.num_classes = 10 if dataset == "CIFAR10" else 100 if dataset == "CIFAR100" else 7 if dataset == "PACS" else 65 if dataset == "OfficeHome" else None

        # Initialize model based on dataset and model type
        if dataset in ["CIFAR10", "CIFAR100", "PACS", "OfficeHome"]:
            if args.model_type == "CNN":
                self.model = CNN(self.num_classes)
            elif args.model_type == "ResNet":
                self.model = ResNet(self.num_classes)
            elif args.model_type == "ViT":
                self.model = ViT_B_32(self.num_classes)
            else:
                raise ValueError(f"Unknown model type: {args.model_type}")
        else:
            raise NotImplementedError(f"Unsupported dataset: {dataset}")

        # Initialize gradients
        self.grad = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Initialize optimizer
        self.init_optimizer(optimizer,learning_rate,weight_decay)

    def init_optimizer(self,optimizer,learning_rate,weight_decay):
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),learning_rate,weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),learning_rate,weight_decay=weight_decay)
    
    def train(self):
        # Create a copy of the global model with the same architecture
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = type(self.model)(self.num_classes) 
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train()
        self.model = self.model.to(self.device)
        
        self.model.zero_grad()

        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                
                y_pred = self.model(batch_x)
                
                loss = self.loss_func(y_pred, batch_y.reshape(-1).long())
                    
                loss.backward()

                self.optimizer.step()   
                
        self.grad = { n:(g-l).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}
        
                
    def update_parameters(self, parameter_dict):
        for params_c,params_global in zip(self.model.parameters(),parameter_dict):
            params_c.data = params_global.clone()