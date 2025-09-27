import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class GradVac():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        self.task_num = 2
        self.k_idx = [-1]
        self.rho_T = torch.zeros(self.task_num, self.task_num, len([-1])).to('cuda')
        self.beta = 0.5
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def ga_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, _ = self._pack_grad(objectives)
        pc_grad = self._project_gradvac(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_gradvac(self, grads):
        grads = torch.stack(grads)
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                for k in range(len(self.k_idx)):
                    beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k+1])
                    if end == -1:
                        end = grads.size()[-1]
                    rho_ijk = torch.dot(pc_grads[tn_i,beg:end], grads[tn_j,beg:end]) / (pc_grads[tn_i,beg:end].norm()*grads[tn_j,beg:end].norm()+1e-8)
                    if rho_ijk < self.rho_T[tn_i, tn_j, k]:
                        w = pc_grads[tn_i,beg:end].norm()*(self.rho_T[tn_i,tn_j,k]*(1-rho_ijk**2).sqrt()-rho_ijk*(1-self.rho_T[tn_i,tn_j,k]**2).sqrt())/(grads[tn_j,beg:end].norm()*(1-self.rho_T[tn_i,tn_j,k]**2).sqrt()+1e-8)
                        pc_grads[tn_i,beg:end] += grads[tn_j,beg:end]*w
                    if rho_ijk > 0:
                        self.rho_T[tn_i,tn_j,k] = (1-self.beta)*self.rho_T[tn_i,tn_j,k] + self.beta*rho_ijk
        new_grads = pc_grads.mean(0)
        return new_grads
    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


