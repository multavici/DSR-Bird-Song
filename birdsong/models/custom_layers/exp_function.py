#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:14:32 2019

@author: ssharma
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# Inherit from Function
class ExpFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=True):
        ctx.save_for_backward(input, weight, bias)
        
        output = torch.pow(input, 1./(1. + torch.exp(-weight))) 
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight) 
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)  
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias
    
class ExpFunctionLayer(nn.Module):
    
    def __init__(self, input_features, output_features, bias=None):
        super(ExpFunctionLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-1.21, -1.73)
        if bias is not None:
            self.bias.data.uniform_( -1.21, -1.73)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        #print('weight==== {}'.format(self.weight.size()))
        return ExpFunction.apply(input, self.weight, self.bias)
    

    def extra_repr(self):
        # (Optional)Set the extra information about this module. 
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
