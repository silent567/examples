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
        mask_sum = torch.sum(mask,dim=dim,keepdim=True)
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
    '''
    grad_input = torch.zeros_like(grad_output)
    unique_output = torch.unique(output)
    for uo in unique_output.unbind():
        mask = output == uo
        grad_input[mask] = torch.sum(grad_output[mask])/torch.sum(mask)

class torch_gfusedlasso(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, A, dim=-1, lam=1.0):
        '''
        input's shape = [*,M,*]
        A's shape = [*,M,M,*]
        '''
        if not torch.is_tensor(lam):
            lam = torch.zeros([],device=input.device,dtype=input.dtype) + lam

        input_size = input.size()
        M = input_size[dim]

        input_reshape = torch.reshape(torch.transpose(input,dim,-1),[-1,M])
        A_reshape = torch.reshape(torch.transpose(torch.transpose(A,dim+1,-1),dim,-2),[-1,M,M])

        output_reshape = torch.stack([torch.from_numpy(gfusedmax(i.numpy(),a.numpy(),lam=lam.numpy()))
                                      for i,a in zip(input_reshape.unbind(),A_reshape.unbind())],dim=0)
        output = torch.transpose(torch.reshape(output_reshape,input_size),dim,-1)

        ctx.dim = dim
        ctx.save_for_backward(input, lam, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, lam, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = torch.zeros_like(grad_output)

        mask = (output > 0).type_as(input)
        masked_grad_output = grad_output*mask
        masked_grad_output -= torch.sum(masked_grad_output,dim=dim,keepdim=True)\
                            / torch.sum(mask,dim=dim,keepdim=True)
        grad_input = masked_grad_output / gamma

        if ctx.needs_input_grad[1]:
            raise ValueError('No gradient is defined for dim argument of sparsemax')

        grad_gamma = None
        if ctx.needs_input_grad[1]:
            grad_gamma = -torch.sum(masked_grad_output*input*mask)/gamma/gamma
        return grad_input, None, grad_gamma

if __name__ == '__main__':
    size = 10
    a = torch.rand(size,requires_grad=True)
    lam = torch.ones([],requires_grad=True)
    torch_sparse = torch_sparsemax.apply(a,-1,lam)
    numpy_sparse = sparsemax(a.detach().numpy(),lam.item())
    torch_sparse.backward(torch.arange(size,dtype=a.dtype))
    print(a,torch_sparse,numpy_sparse)
    print(a.grad, lam.grad)
