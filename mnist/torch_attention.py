#!/usr/bin/env python
# coding=utf-8

from torch_mapping import torch_sparsemax
import torch

# haven't tested
class AddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size=None,query_size=0,projection_flag=True,mapping_func = torch.nn.functional.softmax):
        if not projection_flag and output_size is not None:
            raise ValueError('if projection_flag=False, output_size should be None')

        super(AddAttention,self).__init__()

        self.query_size = query_size
        self.projection_flag = projection_flag
        self.mapping_func = mapping_func

        self.score_func = torch.nn.Linear(input_size+query_size,1)
        if self.projection_flag:
            self.proj_func = torch.nn.Linear(input_size,output_size)

    def forward(self,x,q=None):
        '''
        x's shape = [N,M,input_size]
        '''
        scores = self.score_func(x if self.query_size < 1 else torch.cat([x,q]))
        weights = self.mapping_func.apply(x,dim=-2)
        if self.projection_flag:
            x = self.proj_func(x)
        output = torch.sum(weights*x,dim=-2)
        return output

class ConvAddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride_size,max_type='softmax',layer_norm_flag=True,lam=1.0,gamma=1.0,query_size=0):
        super(ConvAddAttention,self).__init__()
        if max_type == 'sparsemax':
            if gamma is None:
                self.register_parameter('gamma',torch.nn.Parameter(torch.ones([],dtype=torch.float,requires_grad=True)))
            else:
                self.gamma = gamma
            self.mapping_func = lambda x,dim: torch_sparsemax.apply(x,dim,self.gamma)
        else:
            self.mapping_func = torch.nn.functional.softmax

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.query_size = query_size

        self.proj_func = torch.nn.Linear(input_size,output_size)
        self.score_func = torch.nn.Linear(input_size+query_size,1)
        if layer_norm_flag:
            self.score_norm = torch.nn.LayerNorm([self.kernel_size*self.kernel_size,1])
        else:
            self.score_norm = lambda x:x
        self.proj_func.to('cuda')
        self.score_func.to('cuda')
    def forward(self,x,q=None):
        '''
        x's shape = [N,C,H,W]
        q's shape = [N,C']
        '''
        N,C,H,W = x.size()

        x = x.reshape([N,H,W,C])
        proj_x = self.proj_func(x)
        score_x = self.score_func(x if self.query_size < 1 else torch.cat([x,q.unsqueeze_(1).unsqueeze_(1).expand(-1,H,W,-1)],dim=-1))
        output = []
        for h in range(0,H-self.kernel_size+1,self.stride_size):
            tmp_output = []
            for w in range(0,W-self.kernel_size+1,self.stride_size):
                scores = torch.reshape(score_x[:,h:h+self.kernel_size,w:w+self.kernel_size,:],[N,-1,1])
                projs = torch.reshape(proj_x[:,h:h+self.kernel_size,w:w+self.kernel_size,:],[N,-1,self.output_size])
                weights = self.mapping_func(self.score_norm(scores),dim=-2)
                tmp_output.append(torch.sum(projs * weights,dim=-2))
            output.append(torch.stack(tmp_output,dim=-1))
        output = torch.stack(output,dim=-2)

        return output
    def to(self,device):
        super(ConvAddAttention,self).to(device)
        self.gamma = self.gamma.to(device)





