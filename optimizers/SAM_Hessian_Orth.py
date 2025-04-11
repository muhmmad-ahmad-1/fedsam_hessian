import torch
import torch.nn.functional as F
from pyhessian import hessian  # Hessian computation

class SAM_Hessian_Orth(torch.optim.Optimizer):
    '''
    Builds on SAM_Hessian optimizer, with Eigen-SAM inspired perturbation.
    Incorporates the component of the top Hessian eigenvector orthogonal to the gradient
    to improve sharpness-aware updates as described in:
    "Explicit Eigenvalue Regularization Improves Sharpness-Aware Minimization"
    (https://arxiv.org/abs/2501.12666).
    '''
    def __init__(self, params, base_optimizer, rho, alpha=0.1, adaptive=False, maxIter=5, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate: {rho}"
        self.max_norm = 10
        self.alpha = alpha  # Eigen-SAM hyperparameter

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM_Hessian_Orth, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
        self.paras = None
        self.maxIter = maxIter

    @torch.no_grad()
    def first_step(self, hessian_comp:hessian):
        grad_norm = 0.0
        grad_vec = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm += p.grad.norm(p=2) ** 2
                    grad_vec.append(p.grad.view(-1))
        grad_norm = grad_norm.sqrt()
        grad_vector = torch.cat(grad_vec)
        grad_unit = grad_vector / (grad_norm + 1e-7)

        # Get top eigenvector
        _, top_eigenvector = hessian_comp.eigenvalues(maxIter=self.maxIter, tol=1e-2)
        v = top_eigenvector[-1]
        v_vector = torch.cat([x.view(-1) for x in v])
        v_unit = v_vector / (v_vector.norm() + 1e-7)

        # Compute parallel and perpendicular components
        v_dot_g = torch.dot(v_unit, grad_unit)
        v_parallel = v_unit * v_dot_g
        v_perp = F.normalize(v_unit - v_parallel, dim=0)

        sign_align = torch.sign(v_dot_g)

        # Compute final update direction γ
        gamma = grad_unit + self.alpha * sign_align * v_perp
        gamma = gamma / (gamma.norm() + 1e-7)  # Re-normalize γ

        # Apply perturbation
        idx = 0
        for group in self.param_groups:
            scale = group["rho"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                numel = p.numel()
                perturb = gamma[idx:idx + numel].view_as(p) * scale
                p.add_(perturb)
                self.state[p]["e_w"] = perturb
                idx += numel

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update=None):
        inputs, labels, loss_func, model = self.paras
        labels = labels.reshape(-1).long()
        hessian_comp = hessian(model, loss_func, data=(inputs, labels), cuda=True)

        # First forward-backward
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step(hessian_comp)

        # Second forward-backward
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        self.second_step()
