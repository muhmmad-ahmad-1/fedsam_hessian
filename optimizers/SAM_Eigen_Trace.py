import torch
import torch.nn.functional as F
from pyhessian import hessian, Trace_Calculator # Trace computation

class SAM_Eigen_Trace(torch.optim.Optimizer):
    '''
    Derived from SAM_Trace and SAM_Hessian, this optimizer combines both second order methods to provide a more comprehensive
    approach to sharpness aware training. It uses both the trace and the maximum eigenvalue of the Hessian as regularization terms,
    allowing for a more nuanced understanding of the Hessian's structure.
    '''
    def __init__(self, params, base_optimizer, rho, adaptive=False,maxIter = 5,lambda_t = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_Eigen_Trace, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None
        self.maxIter = maxIter
        self.lambda_t = lambda_t    
        

    @torch.no_grad()
    def first_step(self,hessian_comp):
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
        predictions = model(inputs)
        hessian_comp = hessian(model,loss_func, data=(inputs, labels), cuda=True)
        # loss = loss_func(predictions, labels) + hessian_comp.trace(maxIter=2)[-1] * 0.1
        loss = loss_func(predictions,labels)
        self.zero_grad()
        loss.backward()

        self.first_step(hessian_comp)

        predictions = model(inputs)
        hessian_comp = Trace_Calculator(model,loss_func, data=(inputs, labels), cuda=True)
        loss = loss_func(predictions, labels) + torch.mean(hessian_comp.compute_hessian_trace(n_samples=self.maxIter)) * self.lambda_t
        self.zero_grad()
        loss.backward()
        self.second_step()

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]), p=2)
        return norm