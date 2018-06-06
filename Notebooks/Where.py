import os
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from Vision import *

minibatch_size = 100  # quantity of examples that'll be processed
lr = 0.05
n_hidden1 = int(((N_theta*N_azimuth*N_eccentricity*N_phase)/4)*3)
n_hidden2 = int(((N_theta*N_azimuth*N_eccentricity*N_phase)/4))

n_hidden1 = 80
n_hidden2 = 200

print('n_hidden1', n_hidden1, ' / n_hidden2', n_hidden2)
verbose = 1
mean, std = 0.13,  .3
mean, std = 0.,  .3


do_cuda = False # torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if do_cuda else {}
device = torch.cuda.device("cuda" if do_cuda else "cpu")

def get_data_loader(mean=mean, std=std, minibatch_size=minibatch_size):
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data',
                       train=True,     # def the dataset as training data
                       download=True,  # download if dataset not present on disk
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize(mean=(mean,), std=(std,))])),
                           transforms.Normalize((0.1307,), (0.3081,))])),
                       batch_size=minibatch_size,
                       shuffle=True, **kwargs)
    return data_loader
data_loader = get_data_loader(mean=mean, std=std, minibatch_size=minibatch_size)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, data, do_leaky_relu=True):
        if not do_leaky_relu:
            data = F.relu(self.hidden1(data))
            data = F.relu(self.hidden2(data))
        else:
            data = F.leaky_relu(self.hidden1(data))
            data = F.leaky_relu(self.hidden2(data))
        data =  F.sigmoid(self.predict(data))
        return data

#print(device)
net = Net(n_feature=N_theta*N_azimuth*N_eccentricity*N_phase, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=N_azimuth*N_eccentricity)
# net = net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
# https://pytorch.org/docs/master/nn.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss
#loss_func = torch.nn.BCEWithLogitsLoss()
loss_func = torch.nn.BCELoss()


def train(net, minibatch_size, optimizer=optimizer, vsize=N_theta*N_azimuth*N_eccentricity*N_phase,
            asize=N_azimuth*N_eccentricity, offset_std=10, offset_max=25, verbose=1, contrast=std):
    t_start = time.time()
    if verbose: print('Starting training...')
    for batch_idx, (data, label) in enumerate(data_loader):
        optimizer.zero_grad()

        input_ = np.zeros((minibatch_size, 1, vsize))
        a_data = np.zeros((minibatch_size, 1, asize))
        # target = np.zeros((minibatch_size, asize))

        for idx in range(minibatch_size):
            i_offset = minmax(np.random.randn()*offset_std, offset_max)
            j_offset = minmax(np.random.randn()*offset_std, offset_max)
            # input_[idx, 0, :], a_data[idx, 0, :] = couples(data[idx, 0, :, :], i_offset, j_offset, contrast=contrast)
            # target[idx, :] = a_data[idx, 0, :]
            input_[idx, 0, :], a_data[idx, 0, :] = couples(data[idx, 0, :, :], i_offset, j_offset, contrast=contrast)
            #target[idx, :] = a_data[idx, 0, :]

        #input_, target = Variable(torch.FloatTensor(input_)), Variable(torch.FloatTensor(a_data))
        input_, a_data = Variable(torch.FloatTensor(input_)), Variable(torch.FloatTensor(a_data))
        #input_, a_data = input_.to(device), a_data.to(device)
        #print('a_data.numpy()', a_data.numpy().shape)
        #print('... min, max=', a_data.numpy().min(), a_data.numpy().max())
        prediction = net(input_)
        #loss = loss_func(prediction, target)
        loss = loss_func(prediction, a_data)


        #input_, target = Variable(torch.FloatTensor(input_)), Variable(torch.FloatTensor(a_data))
        # input_, target = Variable(torch.FloatTensor(input_)), Variable(torch.FloatTensor(target))
        # data, target = data.to(self.device), target.to(self.device)

        # prediction = net(input_)
        # loss = loss_func(prediction, target)

        loss.backward()
        optimizer.step()

        if verbose and batch_idx % 100 == 0:
            print('[{}/{}] Loss: {} Time: {:.2f} mn'.format(
                batch_idx*minibatch_size, len(data_loader.dataset),
                loss.data.numpy(), (time.time()-t_start)/60))
    return net


def test(net, minibatch_size, optimizer=optimizer, vsize=N_theta*N_azimuth*N_eccentricity*N_phase, asize=N_azimuth*N_eccentricity, offset_std=10, offset_max=25):
    for batch_idx, (data, label) in enumerate(data_loader):
        input_, a_data = np.zeros((minibatch_size, 1, vsize)), np.zeros(
            (minibatch_size, 1, asize))
        target = np.zeros((minibatch_size, asize))
        for idx in range(minibatch_size):

            i_offset, j_offset = minmax(np.random.randn()*offset_std, offset_max), minmax(np.random.randn()*offset_std, offset_max)
            input_[idx, 0, :], a_data[idx, 0, :] = couples(data[idx, 0, :], i_offset, j_offset)
            target[idx, :] = a_data[idx, 0, :]

        input_ = Variable(torch.FloatTensor(input_))
        target = Variable(torch.FloatTensor(a_data))

        prediction = net(input_)
        loss = loss_func(prediction, target)

    return loss.data.numpy()


def eval_sacc(vsize=N_theta*N_azimuth*N_eccentricity*N_phase, asize=N_azimuth*N_eccentricity,
             N_pic=N_X, sacc_lim=5, fovea_size=10, offset_std=10, offset_max=25, fig_type='cmap'):
    for batch_idx, (data, label) in enumerate(data_loader):
        #data = data.to(device)
        i_offset = minmax(np.random.randn()*offset_std, offset_max)
        j_offset = minmax(np.random.randn()*offset_std, offset_max)
        print('Stimulus position: ({},{})'.format(i_offset, j_offset))
        # a_data_in_fovea = False
        # sacc_count = 0

        if True: #while not a_data_in_fovea:
            input_, a_data = np.zeros((1, 1, vsize)), np.zeros((1, 1, asize))
            input_[0, 0, :], a_data[0, 0, :] = couples(data[0, 0, :], i_offset, j_offset)
            #input_, a_data = Variable(torch.FloatTensor(input_)), Variable(torch.FloatTensor(a_data))

            input_ = Variable(torch.FloatTensor(input_))
            a_data = Variable(torch.FloatTensor(a_data))

            prediction = net(input_)
            pred_data = prediction.data.numpy()[-1][-1]

            if fig_type == 'cmap':
                image = colliculus_inverse @ pred_data
                image_reshaped = image.reshape(N_pic, N_pic)

                fig, ax = plt.subplots(figsize=(13, 10.725))
                cmap = ax.pcolor(np.arange(-(N_pic/2), (N_pic/2)),
                                 np.arange(-(N_pic/2), (N_pic/2)), image_reshaped)
                fig.colorbar(cmap)
                plt.axvline(j_offset, c='k')
                plt.axhline(i_offset, c='k')

                for i_pred in range(0, N_pic):
                    for j_pred in range(0, N_pic):
                        if image_reshaped[i_pred][j_pred] == image_reshaped.max():
                            i_hat, j_hat = i_pred-(N_pic/2), j_pred-(N_pic/2)
                            print('Position prediction: ({},{})'.format(
                                i_hat, j_hat))
                            if fig_type == 'cmap':
                                plt.axvline(j_hat, c='r')
                                plt.axhline(i_hat, c='r')
                            break

            if fig_type == 'log':
                # print(pred_data.shape)
                print('Loading pred_data... min, max=', pred_data.min(), pred_data.max())

                # code = colliculus_inverse @ pred_data
                # global_colliculus = colliculus_vector @ code
                global_colliculus = pred_data.reshape(N_eccentricity, N_azimuth)

                log_r_a_data = 1 + \
                    np.log(np.sqrt(i_offset**2 + j_offset**2) /
                           np.sqrt(N_X**2 + N_Y**2) / 2) / 5
                if j_offset != 0:
                    azimuth_a_data = np.arctan(-i_offset / j_offset)
                else:
                    azimuth_a_data = np.sign(-i_offset) * np.pi/2
                print('a_data position (log_r, azimuth) = ({},{})'.format(log_r_a_data,
                                                                        azimuth_a_data))
                azimuth, log_r = np.meshgrid(np.linspace(-np.pi, np.pi, N_azimuth), np.linspace(0, 1, N_eccentricity))
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                #ax.imshow(np.fliplr(global_colliculus))
                # ax.pcolormesh(np.fliplr(global_colliculus))
                cmap = ax.pcolor(log_r, azimuth, global_colliculus)
                ax.plot(azimuth_a_data, log_r_a_data, 'r+')
                fig.colorbar(cmap)



        print('*' * 50)
        return prediction



def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        Accuracy = test()
    print('Test set: Final Accuracy: {:.3f}%'.format(Accuracy*100)) # print que le pourcentage de réussite final


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', #default = 0.5
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dimension', type=int, default = 120, metavar='D',
                        help='the dimension of the second neuron network') #ajout de l'argument dimension représentant le nombre de neurone dans la deuxième couche.
    parser.add_argument('--boucle', type=int, default=0, metavar='B',
                       help='boucle pour faire différents couche de la deuxième couche de neurone')# ajout de boucle pour automatiser le nombre de neurone dans la deuxieme couche
    args = parser.parse_args()

    if args.boucle == 1: # Pour que la boucle se fasse indiquer --boucle 1
        rho = 10**(1/3)
        for i in [int (k) for k in rho**np.arange(2,9)]:# i prend les valeur en entier du tuple rho correspondra au nombre de neurone
            args.dimension = i
            print('La deuxième couche de neurone comporte',i,'neurones')
            main()
    else:
        t0 = time.time () # ajout de la constante de temps t0

        main()

        t1 = time.time () # ajout de la constante de temps t1

        print("Le programme a mis",t1-t0, "secondes à s'exécuter.") #compare t1 et t0, connaitre le temps d'execution du programme
