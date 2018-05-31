import os
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
# import noise

# TODO: passer les arguments par la ligne de commande
N_theta, N_azimuth, N_eccentricty, N_phase, N_X, N_Y, rho = 6, 12, 8, 2, 128, 128, 1.41
minibatch_size = 100  # quantity of examples that'll be processed
lr = 0.05
n_hidden1 = int(((N_theta*N_azimuth*N_eccentricty*N_phase)/4)*3)
n_hidden2 = int(((N_theta*N_azimuth*N_eccentricty*N_phase)/4))
verbose = 1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import MotionClouds as mc
from torchvision import datasets, transforms
from torch.autograd import Variable
from LogGabor import LogGabor


## Charger la matrice de certitude
path = "MNIST_accuracy.npy"
if os.path.isfile(path):
    accuracy =  np.load(path)
    if verbose:
        print('Loading accuracy... min, max=', accuracy.min(), accuracy.max())
else:
    print('No accuracy data found.')

## Préparer l'apprentissage et les fonctions nécessaires au fonctionnement du script
def vectorization(N_theta, N_azimuth, N_eccentricty, N_phase, N_X, N_Y, rho,
                  ecc_max=.8, B_sf=.4, B_theta=np.pi/N_theta/2):
    retina = np.zeros((N_theta, N_azimuth, N_eccentricty, N_phase, N_X*N_Y))
    parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    lg = LogGabor(parameterfile)
    lg.set_size((N_X, N_Y))
    # params = {'sf_0': .1, 'B_sf': lg.pe.B_sf,
    #           'theta': np.pi * 5 / 7., 'B_theta': lg.pe.B_theta}
    # phase = np.pi/4
    # edge = lg.normalize(lg.invert(lg.loggabor(
    #     N_X/3, 3*N_Y/4, **params)*np.exp(-1j*phase)))

    for i_theta in range(N_theta):
        for i_azimuth in range(N_azimuth):
            for i_eccentricty in range(N_eccentricty):
                ecc = ecc_max * (1/rho)**(N_eccentricty - i_eccentricty)
                r = np.sqrt(N_X**2+N_Y**2) / 2 * ecc  # radius
                sf_0 = 0.5 * 0.03 / ecc
                x = N_X/2 + r * \
                    np.cos((i_azimuth+(i_eccentricty % 2)*.5)*np.pi*2 / N_azimuth)
                y = N_Y/2 + r * \
                    np.sin((i_azimuth+(i_eccentricty % 2)*.5)*np.pi*2 / N_azimuth)
                for i_phase in range(N_phase):
                    params = {'sf_0': sf_0, 'B_sf': B_sf,
                              'theta': i_theta*np.pi/N_theta, 'B_theta': B_theta}
                    phase = i_phase * np.pi/2
                    # print(r, x, y, phase, params)

                    retina[i_theta, i_azimuth, i_eccentricty, i_phase, :] = lg.normalize(
                        lg.invert(lg.loggabor(x, y, **params)*np.exp(-1j*phase))).ravel()
    return retina


retina = vectorization(N_theta, N_azimuth, N_eccentricty, N_phase, N_X, N_Y, rho)
retina_vector = retina.reshape((N_theta*N_azimuth*N_eccentricty*N_phase, N_X*N_Y))
retina_inverse = np.linalg.pinv(retina_vector)

colliculus = (retina**2).sum(axis=(0, 3))
colliculus = colliculus**.5
colliculus /= colliculus.sum(axis=-1)[:, :, None]
colliculus_vector = colliculus.reshape((N_azimuth*N_eccentricty, N_X*N_Y))
colliculus_inverse = np.linalg.pinv(colliculus_vector)


def mnist_fullfield(data, i_offset, j_offset, N_pic=N_X, noise=0., figure_type=''):
    N_stim = data.shape[0]
    center = (N_pic-N_stim)//2

    data_fullfield = (data.min().numpy()) * np.ones((N_pic, N_pic))
    data_fullfield[int(center+i_offset):int(center+N_stim+i_offset), int(center+j_offset):int(center+N_stim+j_offset)] = data

    if noise>0.:
        data_fullfield += noise * MotionCloudNoise()

    data_retina = retina_vector @ np.ravel(data_fullfield)

    if figure_type == 'cmap':
        image_hat = phi_plus @ data_LP
        fig, ax = plt.subplots(figsize=(13, 10.725))
        cmap = ax.pcolor(np.arange(-N_pic/2, N_pic/2), np.arange(-N_pic/2, N_pic/2), image_hat.reshape((N_X, N_X)))
        fig.colorbar(cmap)
        return fig, ax

    elif figure_type == 'log':
        code = phi @ np.ravel(data_LP)
        global_energy = (code**2).sum(axis=(0, -1))
        print(code.shape, global_energy.shape)

        log_r_target = 1 + np.log(np.sqrt(i_offset**2 + j_offset**2) / np.sqrt(N_X**2 + N_Y**2) / 2) / 5
        if j_offset != 0:
            theta_target = np.arctan(-i_offset / j_offset)
        else:
            theta_target = np.sign(-i_offset) * np.pi/2
        log_r, theta = np.meshgrid(np.linspace(0, 1, N_scale+1), np.linspace(-np.pi*.625, np.pi*1.375, N_orient+1))

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.pcolor(theta, log_r, np.fliplr(global_energy))
        ax.plot(theta_target, log_r_target, 'r+')
        return fig, ax
    else:
        return data_retina


def accuracy_fullfield(accuracy, i_offset, j_offset, N_pic=N_X):
    N_stim = accuracy.shape[0]
    center = (N_pic-N_stim)//2

    accuracy_fullfield = 0.1 * np.ones((N_pic, N_pic))
    accuracy_fullfield[int(center+i_offset):int(center+N_stim+i_offset),
                 int(center+j_offset):int(center+N_stim+j_offset)] = accuracy

    accuracy_colliculus = colliculus_vector @ np.ravel(accuracy_fullfield)
    # if verbose: print('accuracy_colliculus... min, max=', accuracy_colliculus.min(), accuracy_colliculus.max())

    return accuracy_colliculus


def couples(data, i_offset, j_offset):#, device):
    #data = data.to(device)
    v = mnist_fullfield(data, i_offset, j_offset)
    a = accuracy_fullfield(accuracy, i_offset, j_offset)
    return (v, a)


def minmax(value, border):
    value = max(value, -border)
    value = min(value, border)
    return int(value)


def MotionCloudNoise(sf_0=0.125, B_sf=3.):
    mc.N_X, mc.N_Y, mc.N_frame = 128, 128, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    name = 'static'
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=.5)

    z = mc.rectif(mc.random_cloud(env))
    z = z.reshape((mc.N_X, mc.N_Y))
    return z


do_cuda = False # torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if do_cuda else {}
device = torch.cuda.device("cuda" if do_cuda else "cpu")


data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/data',
                   train=True,     # def the dataset as training data
                   download=True,  # download if dataset not present on disk
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=minibatch_size,
    shuffle=True, **kwargs)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, data, do_leaky_relu=False):
        if do_leaky_relu:
            data = F.relu(self.hidden1(data))
            data = F.relu(self.hidden2(data))
        else:
            data = F.leaky_relu(self.hidden1(data))
            data = F.leaky_relu(self.hidden2(data))
        data = self.predict(data)
        return F.sigmoid(data)

#print(device)
net = Net(n_feature=N_theta*N_azimuth*N_eccentricty*N_phase, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=N_azimuth*N_eccentricty)
#net = net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
# https://pytorch.org/docs/master/nn.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss
loss_func = torch.nn.BCEWithLogitsLoss()


def train(net, minibatch_size, optimizer=optimizer, vsize=N_theta*N_azimuth*N_eccentricty*N_phase, asize=N_azimuth*N_eccentricty, offset_std=10, offset_max=25, verbose=1):
    t_start = time.time()
    if verbose: print('Starting training...')
    for batch_idx, (data, label) in enumerate(data_loader):

        input = np.zeros((minibatch_size, 1, vsize))
        a_data = np.zeros((minibatch_size, 1, asize))

        target = np.zeros((minibatch_size, asize))
        for idx in range(minibatch_size):

            i_offset = minmax(np.random.randn()*offset_std, offset_max)
            j_offset = minmax(np.random.randn()*offset_std, offset_max)
            input[idx, 0, :], a_data[idx, 0, :] = couples(data[idx, 0, :, :], i_offset, j_offset)
            target[idx, :] = a_data[idx, 0, :]

        input, target = Variable(torch.FloatTensor(input)), Variable(torch.FloatTensor(a_data))

        prediction = net(input)
        loss = loss_func(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and batch_idx % 100 == 0:
            print('[{}/{}] Loss: {} Time: {:.2f} mn'.format(
                batch_idx*minibatch_size, len(data_loader.dataset),
                loss.data.numpy(), (time.time()-t_start)/60))
    return net


def test(net, minibatch_size, optimizer=optimizer, vsize=N_theta*N_azimuth*N_eccentricty*N_phase, asize=N_azimuth*N_eccentricty, offset_std=10, offset_max=25):
    for batch_idx, (data, label) in enumerate(data_loader):
        input, a_data = np.zeros((minibatch_size, 1, vsize)), np.zeros(
            (minibatch_size, 1, asize))
        target = np.zeros((minibatch_size, asize))
        for idx in range(minibatch_size):

            i_offset, j_offset = minmax(np.random.randn()*offset_std, offset_max), minmax(np.random.randn()*offset_std, offset_max)
            input[idx, 0, :], a_data[idx, 0, :] = couples(data[idx, 0, :], i_offset, j_offset)
            target[idx, :] = a_data[idx, 0, :]

        input = Variable(torch.FloatTensor(input))
        target = Variable(torch.FloatTensor(a_data))

        prediction = net(input)
        loss = loss_func(prediction, target)

    return loss.data.numpy()


def eval_sacc(vsize=N_theta*N_azimuth*N_eccentricty*N_phase, asize=N_azimuth*N_eccentricty,
             N_pic=N_X, sacc_lim=5, fovea_size=10, offset_std=10, offset_max=25, fig_type='cmap'):
    for batch_idx, (data, label) in enumerate(data_loader):
        #data = data.to(device)
        i_offset = minmax(np.random.randn()*offset_std, offset_max)
        j_offset = minmax(np.random.randn()*offset_std, offset_max)
        print('Stimulus position: ({},{})'.format(i_offset, j_offset))
        # a_data_in_fovea = False
        # sacc_count = 0

        if True: #while not a_data_in_fovea:
            input, a_data = np.zeros((1, 1, vsize)), np.zeros((1, 1, asize))
            input[0, 0, :], a_data[0, 0, :] = couples(data[0, 0, :], i_offset, j_offset)
            #input, a_data = Variable(torch.FloatTensor(input)), Variable(torch.FloatTensor(a_data))

            input = Variable(torch.FloatTensor(input))
            a_data = Variable(torch.FloatTensor(a_data))

            prediction = net(input)
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

                # # check if number of saccades is beyond threshold
                # if sacc_count == sacc_lim:
                #     print('Stimulus position not found, break')
                #     break
                #
                # # saccades
                # i_offset, j_offset = (i_offset - i_hat), (j_offset - j_hat)
                # sacc_count += 1
                # print('Stimulus position after saccade: ({}, {})'.format(i_offset, j_offset))
                #
                # # check if the image position is predicted within the fovea
                # if i_hat <= (fovea_size/2) and j_hat <= (fovea_size/2):
                #     if i_hat >= -(fovea_size/2) and j_hat >= -(fovea_size/2):
                #         a_data_in_fovea = True
                #         print('a_data predicted in fovea, stopping the saccadic exploration')

            if fig_type == 'log':
                print(pred_data.shape)
                print('Loading pred_data... min, max=', pred_data.min(), pred_data.max())

                # code = colliculus_inverse @ pred_data
                # global_colliculus = colliculus_vector @ code
                global_colliculus = pred_data.reshape(N_eccentricty, N_azimuth)

                log_r_a_data = 1 + \
                    np.log(np.sqrt(i_offset**2 + j_offset**2) /
                           np.sqrt(N_X**2 + N_Y**2) / 2) / 5
                if j_offset != 0:
                    azimuth_a_data = np.arctan(-i_offset / j_offset)
                else:
                    azimuth_a_data = np.sign(-i_offset) * np.pi/2
                print('a_data position (log_r, azimuth) = ({},{})'.format(log_r_a_data,
                                                                        azimuth_a_data))
                azimuth, log_r = np.meshgrid(np.linspace(-np.pi, np.pi, N_azimuth), np.linspace(0, 1, N_eccentricty))
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                #ax.imshow(np.fliplr(global_colliculus))
                # ax.pcolormesh(np.fliplr(global_colliculus))
                cmap = ax.pcolor(log_r, azimuth, global_colliculus)
                ax.plot(azimuth_a_data, log_r_a_data, 'r+')
                fig.colorbar(cmap)
                #
                # for i_azimuth in range(N_azimuth):
                #     for i_eccentricty in range(N_eccentricty):
                #         if global_colliculus[i_azimuth][i_eccentricty] == np.max(global_colliculus):
                #             print('Position prediction (orient, scale) = ({},{})'.format(
                #                 i_azimuth, i_eccentricty))

                # a_data_in_fovea = True


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
