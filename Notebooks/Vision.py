import os
import time
import matplotlib.pyplot as plt
import numpy as np
import MotionClouds as mc
from LogGabor import LogGabor

# TODO: passer les arguments par la ligne de commande
N_theta, N_azimuth, N_eccentricity, N_phase, N_X, N_Y, rho = 6, 12, 8, 2, 128, 128, 1.41
verbose = 1

## Charger la matrice de certitude
path = "MNIST_accuracy.npy"
if os.path.isfile(path):
    accuracy =  np.load(path)
    if verbose:
        print('Loading accuracy... min, max=', accuracy.min(), accuracy.max())
else:
    print('No accuracy data found.')

## Préparer l'apprentissage et les fonctions nécessaires au fonctionnement du script
def vectorization(N_theta=N_theta, N_azimuth=N_azimuth, N_eccentricity=N_eccentricity, N_phase=N_phase, N_X=N_X, N_Y=N_Y, rho=rho, ecc_max=.8, B_sf=.4, B_theta=np.pi/N_theta/2, figure_type='', save=False):
    retina = np.zeros((N_theta, N_azimuth, N_eccentricity, N_phase, N_X*N_Y))
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
            for i_eccentricity in range(N_eccentricity):
                ecc = ecc_max * (1/rho)**(N_eccentricity - i_eccentricity)
                r = np.sqrt(N_X**2+N_Y**2) / 2 * ecc  # radius
                sf_0 = 0.5 * 0.03 / ecc
                x = N_X/2 + r * \
                    np.cos((i_azimuth+(i_eccentricity % 2)*.5)*np.pi*2 / N_azimuth)
                y = N_Y/2 + r * \
                    np.sin((i_azimuth+(i_eccentricity % 2)*.5)*np.pi*2 / N_azimuth)
                for i_phase in range(N_phase):
                    params = {'sf_0': sf_0, 'B_sf': B_sf,
                              'theta': i_theta*np.pi/N_theta, 'B_theta': B_theta}
                    phase = i_phase * np.pi/2
                    # print(r, x, y, phase, params)

                    retina[i_theta, i_azimuth, i_eccentricity, i_phase, :] = lg.normalize(
                        lg.invert(lg.loggabor(x, y, **params)*np.exp(-1j*phase))).ravel()
    if figure_type == 'retina':
        FIG_WIDTH = 10
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH))
        for i_theta in range(N_theta):
            for i_azimuth in range(N_azimuth):
                for i_eccentricity in range(N_eccentricity):
                    env = np.sqrt(retina[i_theta, i_azimuth, i_eccentricity, 0, :]**2 + retina[i_theta, i_azimuth, i_eccentricity, 1, :]**2).reshape((N_X, N_Y))
                    ax.contourf(env, levels=[env.max()/1.2, env.max()/1.00001], lw=1, colors=[plt.cm.viridis(i_theta/(N_theta))], alpha=.1)
        fig.suptitle('Tiling of visual space using the retinal filters')
        ax.set_xlabel(r'$Y$')
        ax.set_ylabel(r'$X$')
        ax.axis('equal')
        if save: plt.savefig('retina_filter.pdf')
        plt.tight_layout()
        return fig, ax
    elif figure_type == 'colliculus':
        FIG_WIDTH = 10
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH))
        for i_azimuth in range(N_azimuth):
            for i_eccentricity in range(N_eccentricity):
                env = np.sqrt(colliculus[i_azimuth, i_eccentricity, :]**2.5).reshape((N_X, N_Y))
                #ax.contour(colliculus[i_azimuth, i_eccentricity, :].reshape((N_X, N_Y)), levels=[env.max()/2], lw=1, colors=[plt.cm.viridis(i_theta/(N_theta))])
                ax.contourf(env, levels=[env.max()/1.2, env.max()/1.00001], lw=1, colors=[plt.cm.viridis(i_eccentricity/(N_eccentricity))], alpha=.1)
        fig.suptitle('Tiling of visual space using energy')
        ax.set_xlabel(r'$Y$')
        ax.set_ylabel(r'$X$')
        ax.axis('equal')
        plt.tight_layout()
        if save: plt.savefig('colliculus_filter.pdf')
        return fig, ax
    else:
        return retina


retina = vectorization(N_theta, N_azimuth, N_eccentricity, N_phase, N_X, N_Y, rho)
retina_vector = retina.reshape((N_theta*N_azimuth*N_eccentricity*N_phase, N_X*N_Y))
retina_inverse = np.linalg.pinv(retina_vector)

colliculus = (retina**2).sum(axis=(0, 3))
colliculus = colliculus**.5
colliculus /= colliculus.sum(axis=-1)[:, :, None]
colliculus_vector = colliculus.reshape((N_azimuth*N_eccentricity, N_X*N_Y))
colliculus_inverse = np.linalg.pinv(colliculus_vector)


def mnist_fullfield(data, i_offset, j_offset, N_pic=N_X, noise=0.,  mean=.25,  std=.25, figure_type='', save=False):
    N_stim = data.shape[0]
    center = (N_pic-N_stim)//2

    data_fullfield = (data.min().numpy()) * np.ones((N_pic, N_pic))
    data_fullfield[int(center+i_offset):int(center+N_stim+i_offset), int(center+j_offset):int(center+N_stim+j_offset)] = data

    if noise>0.:
        data_fullfield += noise * MotionCloudNoise()

    data_retina = retina_vector @ np.ravel(data_fullfield)

    # data normalization
    #data_retina -= data_retina.mean()
    #data_retina /= data_retina.std()
    #data_retina *= std
    #data_retina += mean

    if figure_type == '128':
        fig, ax = plt.subplots(figsize=(13, 10.725))
        cmap = ax.pcolor(np.arange(-N_pic/2, N_pic/2), np.arange(-N_pic/2, N_pic/2), np.flipud(data_fullfield), cmap='Greys_r')
        fig.colorbar(cmap)
        if save: plt.savefig('mnist_128.pdf')
        return fig, ax

    elif figure_type == 'cmap':
        image_hat = retina_inverse @ data_retina
        fig, ax = plt.subplots(figsize=(13, 10.725))
        cmap = ax.pcolor(np.arange(-N_pic/2, N_pic/2), np.arange(-N_pic/2, N_pic/2), np.flipud(image_hat.reshape((N_X, N_X))), cmap='Greys_r')
        fig.colorbar(cmap)
        if save and noise>0.: plt.savefig('mnist_128_LP_noise.pdf')
        elif save and noise==0.: plt.savefig('mnist_128_LP_nonoise.pdf')
        return fig, ax

    elif figure_type == 'log':
        #TODO: les lignes suivantes sont pas 100% claires - a tester dans 2018-05-31_LogPol_figures
        #code = retina_vector @ np.ravel(data_retina)
        global_energy = (data_retina**2).reshape((N_theta*N_azimuth, N_eccentricity*N_phase, N_X*N_Y))
        global_energy = global_energy.sum(axis=0).reshape(N_eccentricity, N_azimuth)
        print(data_retina.shape, global_energy.shape)

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



def accuracy_fullfield(accuracy, i_offset, j_offset, N_pic=N_X, figure_type='', save=False):
    N_stim = accuracy.shape[0]
    center = (N_pic-N_stim)//2

    accuracy_fullfield = 0.1 * np.ones((N_pic, N_pic))
    accuracy_fullfield[int(center+i_offset):int(center+N_stim+i_offset),
                 int(center+j_offset):int(center+N_stim+j_offset)] = accuracy

    accuracy_colliculus = colliculus_vector @ np.ravel(accuracy_fullfield)

    if figure_type == 'colliculus':
        image_hat = colliculus_inverse @ np.ravel(accuracy_colliculus)
        fig, ax = plt.subplots(figsize=(13, 10.725))
        cmap = ax.pcolor(np.arange(-N_pic/2, N_pic/2), np.arange(-N_pic/2, N_pic/2), np.flipud(image_hat.reshape((N_X, N_Y))), cmap='Greys_r')
        fig.colorbar(cmap)
        if save: plt.savefig('accuracy_colliculus.pdf')
        return fig, ax
    else:
        return accuracy_colliculus


def couples(data, i_offset, j_offset): #, device):
    #data = data.to(device)
    v = mnist_fullfield(data, i_offset, j_offset)
    a = accuracy_fullfield(accuracy, i_offset, j_offset)
    return (v, a)


def minmax(value, border):
    value = max(value, -border)
    value = min(value, border)
    return int(value)


def MotionCloudNoise(sf_0=0.125, B_sf=3., figure_type='', save=False):
    mc.N_X, mc.N_Y, mc.N_frame = 128, 128, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    name = 'static'
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=.5)
    z = mc.rectif(mc.random_cloud(env))
    z = z.reshape((mc.N_X, mc.N_Y))

    if figure_type == 'cmap':
        fig, ax = plt.subplots(figsize=(13, 10.725))
        cmap = ax.pcolor(np.arange(-mc.N_X/2, mc.N_X/2), np.arange(-mc.N_X/2, mc.N_X/2), MotionCloudNoise(), cmap='Greys_r')
        fig.colorbar(cmap)
        if save: plt.savefig('motioncloud_noise.pdf')
        return fig, ax
    else:
        return z
