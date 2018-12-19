#!/usr/bin/env python
# coding=utf-8

import torch
from mapping import sparsemax, gfusedlasso

class torch_sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim=-1, gamma=1.0):
        if not torch.is_tensor(gamma):
            gamma = torch.zeros([],device=inp.device,dtype=inp.dtype) + gamma

        reshape_size = [1]*len(inp.size())
        reshape_size[dim] = -1

        inp_div = inp / gamma
        inp_sorted,_ = torch.sort(inp_div, dim=dim, descending=True)
        cumsum = torch.cumsum(inp_sorted,dim=dim)
        mask = (1+torch.arange(1,inp_div.size()[dim]+1,device=inp.device,dtype=torch.float)
                .reshape(reshape_size)*inp_sorted) > cumsum
        mask = mask.type_as(inp)
        tau = (torch.sum(inp_sorted*mask,dim=dim,keepdim=True)-1)/torch.sum(mask,dim=dim,keepdim=True,dtype=torch.float)
        output = torch.clamp(inp-tau,min=0)

        ctx.dim = dim
        ctx.save_for_backward(inp, gamma, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, gamma, output = ctx.saved_tensors
        dim = ctx.dim
        # print(ctx.needs_input_grad,gamma.device)

        mask = (output > 0).type_as(inp)
        masked_grad_output = grad_output*mask
        # print('masked_grad_out:',masked_grad_output.size(),masked_grad_output.dtype,masked_grad_output.device,masked_grad_output.norm().item(),masked_grad_output.std().item())
        # mask_sum = torch.sum(mask,dim=dim,keepdim=True)
        # print('mask_sum:       ',mask_sum.size(),mask_sum.dtype,mask_sum.device,mask_sum.norm().item(),mask_sum.std().item())
        masked_grad_output -= mask * (torch.sum(masked_grad_output,dim=dim,keepdim=True)\
                            / (torch.sum(mask,dim=dim,keepdim=True)+1e-5))

        grad_inp = None
        if ctx.needs_input_grad[0]:
            grad_inp = masked_grad_output / gamma
        if len(ctx.needs_input_grad) < 2:
            return grad_inp

        if ctx.needs_input_grad[1]:
            raise ValueError('No gradient is defined for dim argument of sparsemax')

        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = -torch.sum(masked_grad_output*inp*mask)/gamma/gamma
        # print('inp:            ',inp.size(),inp.dtype,inp.device,inp.norm().item(),inp.std().item())
        # print('output:         ',output.size(),output.dtype,output.device,output.norm().item(),output.std().item())
        # print('mask:           ',mask.size(),mask.dtype,mask.device,mask.norm().item(),mask.std().item())
        # print('masked_grad_out:',masked_grad_output.size(),masked_grad_output.dtype,masked_grad_output.device,masked_grad_output.norm().item(),masked_grad_output.std().item())
        # print('grad_out:       ',grad_output.size(),grad_output.dtype,grad_output.device,grad_output.norm().item(),grad_output.std().item())
        # print('grad_inp:       ',grad_inp.size(),grad_inp.dtype,grad_inp.device,grad_inp.norm().item(),grad_inp.std().item())
        # if input() == 'exit':
            # raise ValueError('Commanded to exit')
        return grad_inp, None, grad_gamma

def backward_gfusedmax_torch_1D(input, A, lam, output, grad_output):
    '''
    input's shape = [d]
    A's shape = [d,d]
    lam's shape = []
    output's shape = [d]
    grad_output's shape = [d]

    return grad_input's shape = [d]
    '''
    grad_input = torch.zeros_like(grad_output)
    unique_output = torch.unique(output)
    for uo in unique_output.unbind():
        mask = output == uo
        grad_input[mask] = torch.sum(grad_output[mask])/torch.sum(mask)

    ##### A and lam's gradients to be implemented 
    return grad_input
    

class torch_gfusedlasso(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, A, dim=-1, lam=1.0):
        '''
        input's shape = [*,M,*]
        A's shape = [*,M,M,*]
        '''
        if not torch.is_tensor(lam):
            lam = torch.zeros([],device=input.device,dtype=input.dtype) + lam
        if A.size()[0] == 1:
            A = A.expand([x.size()[0],]+[-1]*(len(A.size())-1))

        M = input.size()[dim]
        input_reshape_size = input.size()

        input_reshape = torch.reshape(torch.transpose(input,dim,-1),[-1,M])
        A_reshape = torch.reshape(torch.transpose(torch.transpose(A,dim+1,-1),dim,-2),[-1,M,M])

        output_reshape = torch.stack([torch.from_numpy(gfusedlasso(i.detach().numpy(),a.detach().numpy(),lam=lam.item()))
                                      for i,a in zip(input_reshape.unbind(),A_reshape.unbind())],dim=0)
        output = torch.transpose(torch.reshape(output_reshape,input_reshape_size),dim,-1)

        ctx.dim, ctx.M, ctx.input_reshape_size = dim, M, input_reshape_size
        ctx.save_for_backward(input_reshape, A_reshape, lam, output_reshape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.needs_input_grad) < 1 or not ctx.needs_input_grad[0]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 3 and ctx.needs_input_grad[3]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')

        input_reshape, A_reshape, lam, output_reshape = ctx.saved_tensors
        dim, M, input_reshape_size  = ctx.dim, ctx.M, ctx.input_reshape_size

        grad_output_reshape = torch.reshape(torch.transpose(grad_output,dim,-1),[-1,M])
        grad_input_reshape = torch.stack([backward_gfusedmax_torch_1D(i,a,lam,o,go) for i,a,o,go in zip(
            input_reshape.unbind(), A_reshape.unbind(), output_reshape.unbind(), grad_output_reshape.unbind()
        )],dim=0)
        grad_input = torch.transpose(torch.reshape(grad_input_reshape,input_reshape_size),dim,-1)
        
        return (grad_input,)+(None)*(len(ctx.needs_input_grad)-1)

class Gfusedmax(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=1.0):
        self.gfusedlasso_func = lambda x,A,dim: torch_gfusedlasso.apply(x,A,dim,lam)
        self.sparsemax_func = lambda x,dim: torch_sparsemax.apply(x,dim,gamma)
    def forward(self,x,A,dim=-1):
        fused_x = self.gfusedlasso_func(x,A,dim)
        output = self.sparsemax_func(fused_x,dim)
        return output

if __name__ == '__main__':
    size = 10
    a = torch.rand(size,requires_grad=True)
    lam = torch.ones([],requires_grad=True)
    torch_sparse = torch_sparsemax.apply(a,-1,lam)
    numpy_sparse = sparsemax(a.detach().numpy(),lam.item())
    torch_sparse.backward(torch.arange(size,dtype=a.dtype))
    print(a,torch_sparse,numpy_sparse)
    print(a.grad, lam.grad)
