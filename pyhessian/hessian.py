#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            eigenvalues, eigenvectors = torch.linalg.eig(T)

            eigen_list = eigenvalues.real
            weight_list = torch.pow(eigenvectors[0,:], 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full


# class Trace_Calculator:
#     def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
#         assert (data is not None and dataloader is None) or (data is None and dataloader is not None)

#         self.model = model.eval()  # Ensure model is in evaluation mode
#         self.criterion = criterion
#         self.device = 'cuda' if cuda else 'cpu'

#         if data is not None:
#             self.data = data
#             self.full_dataset = False
#         else:
#             self.data = dataloader
#             self.full_dataset = True

#         if not self.full_dataset:
#             self.inputs, self.targets = self.data
#             self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(self.device)

#             # Compute gradients for single-batch case
#             outputs = self.model(self.inputs)
#             loss = self.criterion(outputs, self.targets)
#             loss.backward(create_graph=True)

#         # Extract parameters and gradients
#         self.params, self.gradsH = self.get_params_grad(self.model)

#     def trace(self, maxIter=100, tol=1e-3):
#         """
#         Compute the trace of the Hessian using Hutchinson's method.
#         maxIter: Maximum iterations
#         tol: Convergence tolerance
#         """

#         trace_vhv = []
#         trace = torch.tensor(0.0, device=self.device, requires_grad=True)

#         for i in range(maxIter):
#             self.model.zero_grad()

#             # Generate Rademacher random vectors with requires_grad=True
#             v = [torch.randint_like(p, high=2, device=self.device, dtype=p.dtype, requires_grad=True) for p in self.params]
#             v = [torch.where(v_i == 0, torch.tensor(-1, device=self.device, dtype=v_i.dtype), v_i).detach().requires_grad_() for v_i in v]

#             if self.full_dataset:
#                 _,Hv = self.dataloader_hv_product(v)
#             else:
#                 Hv = self.hessian_vector_product(self.gradsH, self.params, v)
#             trace_estimate = self.group_product(Hv, v)
#             trace_vhv.append(trace_estimate)

#             # Compute moving mean of trace estimates
#             trace = torch.mean(torch.stack(trace_vhv))

#             # Convergence check
#             if len(trace_vhv) > 1 and abs(trace - torch.mean(torch.stack(trace_vhv[:-1]))) / (abs(trace) + 1e-6) < tol:
#                 break

#         return torch.stack(trace_vhv)  # Ensure output retains gradients

#     # Function Updates
#     def group_product(self,xs, ys):
#         """ Compute the inner product of two lists of tensors """
#         return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

#     def get_params_grad(self,model):
#         """ Get model parameters and their gradients """
#         params = []
#         grads = []
#         for param in model.parameters():
#             if not param.requires_grad:
#                 continue
#             params.append(param)
#             grads.append(param.grad if param.grad is not None else torch.zeros_like(param))
#         return params, grads

#     # Ensure Hessian-Vector Product (Hv) is differentiable
#     def hessian_vector_product(self,gradsH, params, v):
#         """ Compute Hessian-vector product (Hv) """
#         hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
#         return hv
    
#     def dataloader_hv_product(self,V):
#         """
#         Compute the Hessian-vector product using the entire dataset
#         """
#         device = self.device
#         num_data = 0
#         THv = [torch.zeros(p.size()).to(device) for p in self.params]
#         for inputs, targets in self.data:
#             self.model.zero_grad()
#             tmp_num_data = inputs.size(0)
#             outputs = self.model(inputs.to(device))
#             loss = self.criterion(outputs, targets.to(device))
#             loss.backward(create_graph=True)
#             params, gradsH = self.get_params_grad(self.model)
#             self.model.zero_grad()
#             Hv = self.hessian_vector_product(gradsH, params, V)
#             THv = [THv1 + Hv1 * float(tmp_num_data) + 0. for THv1, Hv1 in zip(THv, Hv)]
#             num_data += float(tmp_num_data)
#         THv = [THv1 / float(num_data) for THv1 in THv]
#         eigenvalue = self.group_product(THv, V)
#         return eigenvalue, THv

class Trace_Calculator:
    '''
    Uses Hutchinson's method to compute the trace of the Hessian (an unbiased stochastic estimator)
    We run for a few samples and take the average to estimate the trace.
    '''
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        self.model = model.eval()
        self.criterion = criterion
        self.device = 'cuda' if cuda else 'cpu'
        self.data = data
        self.dataloader = dataloader
        if data is not None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if not self.full_dataset:
            self.inputs, self.targets = self.data
            self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(self.device)

            # Compute gradients for single-batch case
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)
        
        else:
            n_batches = 0
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward(create_graph=True)
                n_batches += 1
            
            # Normalize gradients by number of batches
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad /= n_batches

        # Extract parameters and gradients
        self.params, self.gradsH = self.get_params_grad(self.model)

    def get_params_grad(self):
        """ Get model parameters and their gradients """
        params = []
        grads = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            params.append(param)
            grads.append(param.grad if param.grad is not None else torch.zeros_like(param))
        return params, grads

    def compute_hessian_trace(self, n_samples = 100, tol = 1e-3):
        trace = torch.tensor(0.0, device=self.params.device)
        for i in range(n_samples):
            # Generate random Rademacher vectors
            zs = [torch.randint(0, 2, p.size(), device=p.device) * 2.0 - 1.0 for p in self.params]
            # Compute Hessian-vector product H z
            h_zs = torch.autograd.grad(self.gradsH, self.params, grad_outputs=zs, create_graph=True)
            # Compute z^T H z
            sample_trace = sum((h_z * z).sum() for h_z, z in zip(h_zs, zs))
            old_trace = trace
            trace = trace * i / (i + 1) + sample_trace / (i + 1)
            if i > 1 and abs(trace - old_trace) / (abs(trace) + 1e-6) < tol:
                break
        return trace
    
    import torch

class Trace_Dropout_Calculator:
    '''
    Uses an implementation of Dropout based variation of Hutchinson's method to accelerate computation.
    Taken from: "Regularizing Deep Neural Networks with Stochastic Estimators of Hessian Trace"
    (https://arxiv.org/abs/2208.05924)
    Original paper did not dive into runtime gains or performance conservation when shifting from original Hutchinson's
    to this dropout based version to reduce computations.
    NOTE: Run a few test runs to ensure it does bring about an imrpovement in trace regularization approach speed without 
    affecting performance gains before including in the optimizers suite.
    '''
    def __init__(self, model, criterion, p=0.05, data=None, dataloader=None, device = 'cuda'):
        self.model = model.eval()
        self.criterion = criterion
        self.device = device
        self.data = data
        self.dataloader = dataloader
        self.p = p

        if data is not None:
            self.inputs, self.targets = data
            self.inputs, self.targets = self.inputs.to(self.device), self.targets.to(self.device)
            self.full_dataset = False

            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)
        else:
            self.full_dataset = True
            n_batches = 0
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward(create_graph=True)
                n_batches += 1

            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad /= n_batches

        self.params, self.gradsH = self.get_params_grad()

    def get_params_grad(self):
        """Get model parameters and their gradients"""
        params = []
        grads = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param)
                grads.append(param.grad if param.grad is not None else torch.zeros_like(param))
        return params, grads

    def sample_q_vector(self, shape, device):
        """Generate Q(p)-distributed vector: ±1 with prob p, 0 with prob 1-2p"""
        probs = torch.rand(shape, device=device)
        mask = probs < 2 * self.p
        signs = torch.randint_like(probs, low=0, high=2) * 2 - 1  # ±1
        return torch.where(mask, signs, torch.zeros_like(signs))

    def compute_trace_dropout(self, maxIter=100, tol=1e-3):
        trace = torch.tensor(0.0, device=self.device)
        for i in range(maxIter):
            # Q(p)-based noise vector for each parameter
            sigmas = [self.sample_q_vector(p.size(), p.device) for p in self.params]

            # v = g · σ (element-wise)
            v = [g * s for g, s in zip(self.gradsH, sigmas)]

            # h = dv/dω
            h = torch.autograd.grad(v, self.params, grad_outputs=[torch.ones_like(vv) for vv in v], retain_graph=True, create_graph=True)

            # t = σ^T h
            t = sum((s * hh).sum() for s, hh in zip(sigmas, h))
            old_trace = trace
            trace = trace * i / (i + 1) + t / (i + 1)

            if i > 1 and abs(trace - old_trace) / (abs(trace) + 1e-6) < tol:
                break

        return trace

    

