from tqdm.auto import tqdm

import numpy as np
np.random.seed(123)

import torch
torch.manual_seed(123)

import random
random.seed(123)

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import argparse

## Reference: Network Structure and Training Code modified from https://github.com/pytorch/examples/tree/main/mnist
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_raw=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        if return_raw:
            return output, x
        else:
            return output

def train(model, device, train_loader, optimizer, epoch, alpha, num_alpha_samples=10):
    model.train()
    for batch_idx, (data_orig, target_orig) in enumerate(train_loader):
        data_orig, target_orig = data_orig.to(device), target_orig.to(device)
        data = torch.repeat_interleave(data_orig, num_alpha_samples, dim=0)
        target = torch.repeat_interleave(target_orig, num_alpha_samples, dim=0)

        data_noise = torch.randn_like(data, dtype=data.dtype) * alpha
        data = data + data_noise
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
        
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Main: Train MNIST on samples perturbed by Gaussian noise, with specified variance alpha^2
def main(gpu_str='0', alpha=0.2, num_alpha_samples=100, num_epochs=10):
    print (np.random.rand())    ## Should be 0.6964691855978616
    print (torch.rand([1]))     ## Should be tensor([0.2961])

    print (f'gpu_str: {gpu_str}, alpha: {alpha}, num_alpha_samples: {num_alpha_samples}')

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Prepare MNIST...')
    transforms_MNIST = transforms.Compose([transforms.ToTensor()])
    MNIST_train = datasets.MNIST('./data', train=True, download=True, transform=transforms_MNIST)
    MNIST_test = datasets.MNIST('./data', train=False, download=True, transform=transforms_MNIST)
    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=64, shuffle=True)

    # Train with inputs perturbed with p_\alpha
    model = Net().to(device)
    base_h_name = 'TMLR_data/mnist_cnn_0_1.pt'

    # The following assertion will fail if the base model is not trained
    # base model can be trained by calling main(alpha = 0, num_alpha_samples = 1)
    assert(os.path.exists(base_h_name) and alpha > 0)

    print (f'Loading Base Model {base_h_name}')
    model.load_state_dict(torch.load(base_h_name))

    save_name = f'TMLR_data/mnist_cnn_{alpha}_{num_alpha_samples}.pt'
    if os.path.exists(save_name):
        print (f'File Exists! Quitting. {save_name}')
        return

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in tqdm(range(1, num_epochs)):
        train(model, device, train_loader, optimizer, epoch, alpha=alpha, num_alpha_samples=num_alpha_samples)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), save_name)

    print (f'Model saved at {save_name}')

# By default this trains a single MNIST model with specified --alpha
# Uncomment 145-159 to train on all alphas over 4 GPUs
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5, help='gaussian alpha for training')
    parser.add_argument('--num_alpha_samples', type=int, default=100)
    parser.add_argument('--gpu_str', type=str, default='2')
    args = parser.parse_args()

    main (gpu_str=args.gpu_str, alpha=args.alpha, num_alpha_samples=args.num_alpha_samples, num_epochs=10)

    # remaining_alpha = []
    # for alpha in np.arange(0.01, 1, 0.01):
    #     alphap = round(alpha, 2)
    #     check_name = f'TMLR_data/mnist_cnn_{alphap}_100.pt'
    #     if not os.path.exists(check_name):
    #         remaining_alpha.append(alphap)
    #         print (check_name)

    # gpus_avail = [0, 1, 2, 3]
    # rot_idx = 0
    # for alpha in remaining_alpha:
    #     main (gpu_str=f'{gpus_avail[rot_idx]}', alpha=alpha, num_alpha_samples=100)
    #     print (f'GPU: {gpus_avail[rot_idx]}, alpha: {alpha}')

    #     rot_idx = (rot_idx + 1) % len(gpus_avail)    
