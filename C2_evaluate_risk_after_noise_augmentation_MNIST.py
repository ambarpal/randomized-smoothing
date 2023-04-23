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
from torchvision import datasets, transforms

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
        if batch_idx % 500 == 0 and batch_idx > 0:
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Given: noisy trained model = h*p_\alpha: X -> [0, 1].
# Output: smoothed model \phi(\phi(model) * p_\beta)

# smoothed_model_raw(x) 
# = \int_{t} \phi(model)(x - t) p_beta(t) dt
# = \int_{t} \phi(model)(x + t) p_beta(t) dt [assuming p_beta is symmetric about 0]
# = E_{v \sim p_beta} \phi(model)(x + v)
# \approx (1 / N) \sum_{v_i \sim p_beta, i = 1...N} \phi(model)(x + v_i)
# = (1 / N) \phi (\sum_{v_i \sim p_beta, i = 1...N} model(x + v_i))
def compute_risk(h, h_alpha, beta, num_beta_samples, test_loader, device):
    h_alpha.eval()
    h.eval()
    num_correct = 0
    delta_sum = 0
    num_test = 0
    
    with torch.no_grad():
        for batch_idx, (data_orig, target_orig) in enumerate(test_loader):
            data_orig, target_orig = data_orig.to(device), target_orig.to(device)
            
            data = torch.repeat_interleave(data_orig, num_beta_samples, dim=0)
            target = torch.repeat_interleave(target_orig, num_beta_samples, dim=0)
            data_noise = torch.randn_like(data, dtype=data.dtype) * beta
            data = data + data_noise
            
            phi_h_alpha = F.one_hot(h_alpha(data).argmax(dim = 1), num_classes=10)
            
            smoothed_raw = torch.zeros((data_orig.shape[0], 10)).to(device)
            smoothed = torch.zeros_like(smoothed_raw, dtype=torch.int64).to(device)
            per_image = torch.split(phi_h_alpha, num_beta_samples, dim=0)
            
            for idx in range(data_orig.shape[0]):
                smoothed_raw[idx] = torch.mean(per_image[idx] * 1.0, axis = 0)
                
            smoothed_preds = smoothed_raw.argmax(dim = 1)
            cur_correct = np.sum(smoothed_raw.argmax(dim = 1).detach().cpu().numpy() == np.array(target_orig.detach().cpu().numpy()))

            num_correct += cur_correct
            num_test += data_orig.shape[0]
            
            delta_sum += torch.sum(smoothed_preds != target_orig).detach().cpu().numpy()
            
    test_acc = num_correct * 100.0 / num_test
    # print (f'Sanity Check. Test Accuracy of Smoothed Classifier: {test_acc}')

    delta_av = delta_sum * 1.0 / num_test
    # print (f'Delta: {delta_sum}, (1/N) Delta: {delta_av}')
    
    return delta_sum, delta_av

## Takes the MNIST model corresponding to each alpha generated from C1_train_noise_augmented_MNIST.py
## and applies randomized smoothing with gaussian variance beta^2.
## The risk of each smoothed classifier is then computed and saved into TMLR_data/smoothing_mnist_plot_hist_risk.npy

# if beta == -1, the above process is repeated for each beta in np.arange(0, 1, 0.1)
def main(gpu_str='0', num_beta_samples=100, beta_=-1):
                                ## If seed == 123
    print (np.random.rand())    ## Should be 0.6964691855978616
    print (torch.rand([1]))     ## Should be tensor([0.2961])

    print (f'gpu_str: {gpu_str}, num_alpha_samples: {num_beta_samples}')

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Prepare MNIST...')
    transforms_MNIST = transforms.Compose([transforms.ToTensor()])
    MNIST_train = datasets.MNIST('./data', train=True, download=True, transform=transforms_MNIST)
    MNIST_test = datasets.MNIST('./data', train=False, download=True, transform=transforms_MNIST)
    train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=64, shuffle=True)

    alpha_range_check = np.arange(0, 1, 0.01)
    beta_range = np.arange(0, 1, 0.1)

    if beta_ >= 0:
        beta_ = round(beta_, 2)
        beta_range = [beta_]

    h = Net().to(device)
    h.load_state_dict(torch.load('TMLR_models/mnist_cnn_0_1.pt'))

    plot_hist = {}
    results_save_fname = 'TMLR_data/smoothing_mnist_plot_hist_risk.npy'
    if beta_ >= 0: 
        results_save_fname = f'TMLR_data/smoothing_mnist_plot_hist_{beta_}_risk.npy'

    if os.path.exists(results_save_fname):
        plot_hist = np.load(results_save_fname, allow_pickle=True).item()

    print (f'Results will be saved at {results_save_fname}')

    for alpha in alpha_range_check:
        alphap = round(alpha, 2)
        fname = f'TMLR_models/mnist_cnn_{alphap}_100.pt'

        if alpha == 0:
            fname = f'TMLR_models/mnist_cnn_0_1.pt'
        
        if os.path.exists(fname):
            print ('Will Read', fname)
    
    for beta in tqdm(beta_range):
        delta_alpha_beta0 = []
        alpha_range = []
        for alpha in tqdm(alpha_range_check):
            alphap = round(alpha, 2)
            fname = f'TMLR_models/mnist_cnn_{alphap}_100.pt'
            
            if alphap == 0:
                fname = f'TMLR_models/mnist_cnn_0_1.pt'
                
            if os.path.exists(fname):
                print (f'Found {fname}')

                h_alpha = Net().to(device)
                h_alpha.load_state_dict(torch.load(fname))
                delta_sum, delta_av = compute_risk(h, h_alpha, beta, num_beta_samples, test_loader, device)
                plot_hist[(alphap, beta)] = (delta_sum, delta_av)
                
                alpha_range.append(alphap)
                delta_alpha_beta0.append(delta_sum)
                
                np.save(results_save_fname, plot_hist, allow_pickle=True)
    
        print (f'Results saved at {results_save_fname}')

# Generate beta-smoothed classifiers for each alpha-smoothed classifier in TMLR_models
# If beta == -1, do the above for all beta in np.arange(0, 1, 0.1) 
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=-1.0, help='gaussian beta for smoothing')
    parser.add_argument('--num_beta_samples', type=int, default=100)
    parser.add_argument('--gpu_str', type=str, default='3')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    print (f'Seed: {args.seed}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    main (args.gpu_str, args.num_beta_samples, beta_=args.beta)