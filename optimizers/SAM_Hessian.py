import torch
import torch.nn.functional as F
from pyhessian import hessian # Hessian computation

class SAM_Hessian(torch.optim.Optimizer):
    '''
    Base Optimizer for Second Order SAM Optimization:
    Adds a second order term as perturbation (i.e. the maximum eigenvalue of the Hessian) to the existing
    first order term (i.e. the gradient) in SAM.
    '''
    def __init__(self, params, base_optimizer, rho, adaptive=False,maxIter = 5, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_Hessian, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        #self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            #group["adaptive"] = adaptive
        self.paras = None
        self.maxIter =  maxIter
        

    @torch.no_grad()
    def first_step(self,hessian_comp:hessian):
        #first order sum 
        grad_norm = 0
        for group in self.param_groups:
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                    grad_norm+=p.grad.norm(p=2)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(maxIter=self.maxIter, tol=1e-2)
        top_eigenvector = top_eigenvector[-1]
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                # original SAM 
                # e_w = p.grad * scale.to(p)
                # ASAM 
                
                e_w=p.grad * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1 + top_eigenvector[idx] * scale)  
                self.state[p]["e_w"] = e_w + top_eigenvector[idx] * scale
                

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0


    def step(self,g_update=None):
        inputs, labels, loss_func, model = self.paras
        labels = labels.reshape(-1).long()
        hessian_comp = hessian(model,loss_func, data=(inputs, labels), cuda=True)
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step(hessian_comp)

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        self.second_step()