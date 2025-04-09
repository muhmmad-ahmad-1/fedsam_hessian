from utils.utils_libs import *
from utils.utils_dataset import *
from sklearn.metrics import accuracy_score,f1_score
from pyhessian import Trace_Calculator, hessian

class Test:
    def __init__(self,dataset,test_x,test_y,model,device):
        
        if dataset == "CIFAR10":
            self.test_x = test_x.reshape(10000, 3, 32, 32)
            self.test_y = test_y.reshape(10000,1)
        
        self.model = model 
        self.device = device
        self.model = self.model.to(self.device)
        self.dataloader = data.DataLoader(Dataset(self.test_x,self.test_y,dataset_name=dataset),32,False)
    
    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in self.dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate Accuracy
        accuracy = accuracy_score(all_labels, all_preds)

        # Calculate trace of the Hessian over test dataset
        hessian_comp = Trace_Calculator(self.model,torch.nn.CrossEntropyLoss(), dataloader=self.dataloader, cuda=True)
        trace  = torch.mean(hessian_comp.trace())
        hessian_comp = hessian(self.model,torch.nn.CrossEntropyLoss(), data=(inputs, labels), cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        top_eigenvalue = top_eigenvalues[-1]
        
        # Calculate F1-Score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return accuracy, f1, trace, top_eigenvalue