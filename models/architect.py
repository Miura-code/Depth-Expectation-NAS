# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# Modifications made by Shun Miura(https://github.com/Miura-code)



""" Architect controls architecture of cell by computing gradients of alpha """
import copy
import torch

from utils.loss import Expected_Depth_Loss_beta


class Architect():
    def __init__(self, net, w_momentum, w_weight_decay):
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
    
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, weights=None):
        """ First Order!
            Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        logits = self.net(val_X)
        losses = self.net.criterion((logits, val_y), updated_weight=weights, detail=True)
        losses[-1].backward()
        
        return losses

        logits_guide = self.v_teacher_net(val_X)
        logits = self.v_net(val_X)
        # hard_loss, soft_loss, loss = self.v_net.criterion(logits, logits_guide, val_y)
        loss = self.v_net.criterion.hard_criteria(logits, val_y)

        # hard_loss, soft_loss, loss = self.v_net.loss(val_X, logits_guide, val_y, return_detail=True)
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]

        with torch.no_grad():
            for alpha, da in zip(self.net.alphas(), dalpha):
                alpha.grad = da

        # return hard_loss, soft_loss, loss
        return loss

    def unrolled_backward_2nd(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Second Order!
            Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calculate w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calculate unrolled loss
        logits_guide = self.v_teacher_net.forward(val_X)
        loss = self.v_net.loss(val_X, logits_guide, val_y)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi * h
    
    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        logits_guide = self.v_teacher_net.forward(trn_X)
        loss = self.net.loss(trn_X, logits_guide, trn_y)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        with torch.no_grad():
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
            
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)
        
    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        logits_guide = self.v_teacher_net.forward(trn_X)
        loss = self.net.loss(trn_X, logits_guide, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas())

        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        logits_guide = self.v_teacher_net.forward(trn_X)
        loss = self.net.loss(trn_X, logits_guide, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas())

        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
    
    def unrolled_backward_betaConstraint(self, trn_X, trn_y, val_X, val_y, xi, w_optim, weights=None):
        """ First Order!
            Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        logits = self.net(val_X)
        if self.arch_criterion_type == "expected":
            losses = self.net.criterion((logits, val_y), ([self.net.alphas(), self.net.betas()]), updated_weight=weights, detail=True)
        elif self.arch_criterion_type == "beta":
            losses = self.net.criterion((logits, val_y), ([self.net.betas()]), updated_weight=weights, detail=True)
        losses[-1].backward()
        
        return losses
