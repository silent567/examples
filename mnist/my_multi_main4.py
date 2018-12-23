from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch_attention import ConvAddAttention
import time


class Net(nn.Module):
    def __init__(self,max_type,layer_norm_flag,lam,gamma,head_cnt):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        # self.conv1 = ConvAddAttention(1, 20, 5, 1, max_type, layer_norm_flag, lam, gamma)
        # self.conv2 = ConvAddAttention(20, 50, 5, 1, max_type, layer_norm_flag, lam, gamma)
        # self.fc1 = nn.Linear(4*4*50, 500)
        self.attention_pool = [ConvAddAttention(50,int(500/head_cnt),11,1,max_type,layer_norm_flag,lam,gamma) for _ in range(head_cnt)]
        for i,att in enumerate(self.attention_pool):
            self.add_module('ConvAddAttention%d'%i,att)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #[26,26]
        x = F.max_pool2d(x, 2, 2) #[13,13]
        x = F.relu(self.conv2(x)) #[11,11]
        # x = F.max_pool2d(x, 2, 2) #[5,5]
        # x = x.view(-1, 4*4*50)
        # x = F.relu(self.fc1(x))
        att_out = torch.cat([att(x) for att in self.attention_pool],dim=1).view(-1,500)
        x = F.relu(att_out)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)

def main(args = None):
    # Training settings
    if args is None:
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')

        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')

        # added by tanghao 20181217
        parser.add_argument('--norm-flag', type=bool, default=False,
                            help='Triggering the Layer Normalization flag for attention scores')
        parser.add_argument('--gamma', type=float, default=None,
                            help='Controlling the sparisty of gfusedmax/sparsemax, the smaller, the more sparse')
        parser.add_argument('--lam', type=float, default=1.0,
                            help='Lambda: Controlling the smoothness of gfusedmax, the larger, the smoother')
        parser.add_argument('--max-type', type=str, default='softmax',choices=['softmax','sparsemax','gfusedmax'],
                            help='mapping function in attention')
        parser.add_argument('--optim-type', type=str, default='SGD',choices=['SGD','Adam'],
                            help='mapping function in attention')
        parser.add_argument('--head-cnt', type=int, default=2, metavar='S', choices=[1,2,4,5,10],
                            help='Number of heads for attention (default: 1)')

        args = parser.parse_args()

    print(args)
    args.seed = time.time()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(args.max_type,args.norm_flag,args.lam,args.gamma,args.head_cnt).to(device)
    if args.optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist4_multi_%2.1f_%.4f_%s_%s_%d_%.2f_%.2f_%d.pt"%(test_acc*100,args.lr,args.optim_type,args.max_type,args.norm_flag,args.lam,0.123 if args.gamma is None else args.gamma,args.head_cnt))

    return test_acc

if __name__ == '__main__':
    main()
