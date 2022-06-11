import csv
import gc
import os
import time
from pathlib import Path

import hlsvdpro
import mat73
import pandas as pd
import scipy
import sio as sio
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.suggest.bohb import TuneBOHB
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, random_split, DataLoader

import torch
import torch.nn as nn
import math
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy.fft as fft
from scipy import stats
from torchsummary import summary

import utils.Jmrui
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
# %%
from Model import Encoder_Model
from utils import watrem, Jmrui


# The Engine class is a class that represents a startegy for traingng and testing.
class Engine():
    def __init__(self, parameters):
        self.parameters = parameters
        self.saving_dir = parameters['parent_root'] + parameters['child_root'] + parameters['version']
        self.epoch_dir =  "epoch/"
        self.loging_dir = parameters['parent_root'] + parameters['child_root']
        self.data_dir = parameters['data_dir']
        self.data_dir_ny = parameters['data_dir_ny']
        self.basis_dir = parameters['basis_dir']
        self.test_data_root = parameters['test_data_root']
        Path(self.saving_dir).mkdir(parents=True, exist_ok=True)
        Path(self.saving_dir+self.epoch_dir).mkdir(parents=True, exist_ok=True)
        self.type = parameters['type']
        if 'dSR' in self.type:
            try:
                self.dSR_refsignal=np.load(os.path.join(self.saving_dir, 'dSR_refsignal' + '.npy'))
            except:
                print("refrence signal for the dSR method is not loaded")
        self.max_epoch = parameters['max_epoch']
        self.batchsize = parameters['batchsize']
        self.numOfSample = parameters['numOfSample'];
        self.t_step = parameters['t_step']
        self.trnfreq = parameters['trnfreq']
        self.nauis = parameters['nauis']
        self.save = parameters['save']
        self.tr = parameters['tr']
        self.lr = parameters['lr']
        if self.lr is None:
            self.lr=1e-3
        self.betas = parameters['betas']
        self.depths = parameters['depths']
        self.ens = parameters['ens']
        self.met_name = parameters['met_name']
        self.BG = parameters["BG"]
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, 4096)
        if self.basis_dir is not None:
            self.basisset = (sio.loadmat(self.basis_dir).get('data'))
            if parameters['basis_conj']:
                self.basisset = np.conj(self.basisset)
        else:
            self.basisset = None
        self.wr = parameters['wr']
        self.data_name = parameters['data_name']
        if self.data_dir is not None:
            try:
                self.dataset = scipy.io.loadmat(self.data_dir).get(self.data_name)
            except:
                self.dataset = mat73.loadmat(self.data_dir).get(self.data_name)
        self.ref_dir = parameters['ref_dir']
        self.ref_conj = parameters['ref_conj']
        if self.ref_dir is not None:
            try:
                self.refsignal = scipy.io.loadmat(self.ref_dir).get(self.data_name)
            except:
                self.refsignal = mat73.loadmat(self.ref_dir).get(self.data_name)
            if self.ref_conj == True:
                self.refsignal = np.conj(self.refsignal)
        self.numOfSig = parameters['numOfSig']
        self.sigLen = parameters['sigLen']
        self.truncSigLen = parameters['truncSigLen']
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, self.sigLen)
        self.t = np.arange(0, self.sigLen) * self.t_step
        self.t = np.expand_dims(self.t, 1)
        self.MM = parameters['MM']
        self.MM_f = parameters['MM_f']
        self.MM_d = np.array(parameters['MM_d'])
        self.MM_a = parameters['MM_a']
        self.MM_plot = parameters['MM_plot']
        self.basis_plot = parameters['basis_plot']
        self.basis_need_shift = parameters['basis_need_shift']
        self.aug_params = parameters['aug_params']
        self.tr_prc = parameters['tr_prc']
        self.in_shape= parameters['in_shape']
        self.enc_type = parameters['enc_type']
        self.banorm = parameters['banorm']
        if self.basis_dir is not None:
            max_c = np.array(parameters['max_c'])
            min_c = np.array(parameters['min_c'])
            self.min_c = (min_c) / np.max((max_c));
            self.max_c = (max_c) / np.max((max_c));
        self.reg_wei = parameters['reg_wei']
        self.data_conj = parameters['data_conj']
        self.test_nos = parameters['test_nos']
        self.quality_filt = parameters['quality_filt']
        self.test_name = parameters['test_name']
        self.beta_step = parameters['beta_step']
        self.MM_type = parameters['MM_type']
        self.MM_dir = parameters['MM_dir']
        self.MM_constr = parameters['MM_constr']
        # TODO it should be soft
        self.comp_freq = (parameters['comp_freq'])
        self.numofnauis = parameters['numofnauis']
        if self.MM_dir is not None:
            self.mm = sio.loadmat(self.MM_dir).get("data")
        self.sim_params = parameters['sim_params']
        if self.sim_params is not None:
            for i, val in enumerate(self.sim_params):
                if isinstance(val,str):
                    self.sim_params[i] = getattr(self, self.sim_params[i])
        if self.MM:
            if parameters['MM_model'] == "lorntz":
                self.MM_model = self.Lornz
                self.MM_d = (np.pi * self.MM_d)
                # self.MM_d = (np.pi ** self.MM_d) * ((self.MM_d) ** 2) / (2 * np.log(2))
            if parameters['MM_model'] == "gauss":
                self.MM_model = self.Gauss
                self.MM_d = (np.pi**2) * ((self.MM_d) ** 2)/(8 * np.log(2))
            self.numOfMM = len(self.MM_f)
            if self.MM_type == 'single' or self.MM_type == 'single_param':
                self.met_name.append("MM")
        self.heatmap_cmap = sns.diverging_palette(20, 220, n=200)

    def savefig(self, tight,path):
        """
        The function takes in a boolean value and a string. If the boolean value is true, it tightens the layout of the
        plot. If the boolean value is false, it does nothing. Then, it saves the plot as a .svg and .png file in the
        directory specified by the string.

        :param tight: if True, the figure will be saved with tight layout
        :param path: the path to the file you want to save
        """
        if tight:
            plt.tight_layout()
        if self.save:
            plt.savefig(self.saving_dir + path + ".svg", format="svg")
            plt.savefig(self.saving_dir + path + " .png", format="png", dpi=800)

    # %%
    def loadModel(autoencoder, path):
        """
        > Loads a model from a given path

        :param autoencoder: the model to be loaded
        :param path: the path to the model file
        :return: The model is being returned.
        """
        # m = LitAutoEncoder(t,signal_norm)
        return autoencoder.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # %%
    def tic(self):
        """
        The function tic() takes no arguments and returns nothing. It sets the global variable start_time to the current
        time
        """
        global start_time
        start_time = time.time()

    def toc(self,name):
        """
        This function takes in a name and a start time, and then prints out the elapsed time since the start time.

        :param name: the name of the file you want to save the time to
        """
        elapsed_time = (time.time() - start_time)
        print("--- %s seconds ---" % elapsed_time)
        timingtxt = open(self.saving_dir + name + ".txt", 'w')
        timingtxt.write(name)
        timingtxt.write("--- %s ----" % elapsed_time)
        timingtxt.close()

    # %%
    def cal_snr(self,data, endpoints=128):
        """
        It takes the first sample of the data and divides it by the standard deviation of the last n points samples of the data

        :param data: the data to be analyzed
        :param endpoints: The number of points to use for the endpoints of the signal, defaults to 128 (optional)
        :return: The signal to noise ratio.
        """
        return np.abs(data[0, :]) / np.std(data.real[-(68 + endpoints):-68, :], axis=0)

    def cal_snrf(self,data_f,endpoints=128):
        """
        It takes the absolute value of the maximum value of the data in the frequency domain, and divides it by the standard
        deviation of the first 128 points of the data in the frequency domain

        :param data_f: the data in the frequency domain
        :param endpoints: The number of points at the beginning of the signal to use for calculating the standard deviation,
        defaults to 128 (optional)
        :return: The maximum value of the absolute value of the data_f array divided by the standard deviation of the real
        part of the data_f array.
        """
        return np.max(np.abs(data_f), 0) / (np.std(data_f.real[:endpoints, :],axis=0))

    def ppm2p(self, r, len):
        """
        It takes the ppm value and converts it to a point in the array

        :param r: the ppm value of the peak
        :param len: the length of the signal
        :return: The ppm value is being converted to a pixel value.
        """
        r = 4.7 - r
        return int(((self.trnfreq * r) / (1 / (self.t_step * len))) + len / 2)

    def ppm2f(self, r):
        """
        The function ppm2f() takes a single argument, r, and returns the value of r multiplied by the value of trnfreq.

        :param r: the number of rotations per minute
        :return: The frequency of the transition in Hz.
        """
        return r * self.trnfreq

    def fillppm(self, y1, y2, ppm1, ppm2, rev, alpha=.1, color='red'):
        """
        This function takes in two spectra, y1 and y2, and fills the area between them with a color of your choice.

        The function takes in the following arguments:

        - y1: The first spectrum
        - y2: The second spectrum
        - ppm1: The ppm value of the first spectrum
        - ppm2: The ppm value of the second spectrum
        - rev: Whether or not to reverse the x-axis
        - alpha: The transparency of the fill
        - color: The color of the fill

        The function then converts the ppm values to points, and then creates a linear space between the two points. It then
        fills the area between the two spectra with the color of your choice.

        The function then checks if the x-axis should be reversed, and if so, it does so.

        :param y1: the first spectrum
        :param y2: the second spectrum
        :param ppm1: The starting point of the region to be filled
        :param ppm2: the ppm value of the right edge of the plot
        :param rev: True if the x-axis is reversed (i.e. ppm values are decreasing from left to right)
        :param alpha: The transparency of the fill
        :param color: The color of the fill, defaults to red (optional)
        """
        p1 = int(self.ppm2p(ppm1, len(y1)))
        p2 = int(self.ppm2p(ppm2, len(y1)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        plt.fill_between(np.flip(x), y1[p2:p1, 0].real,
                         y2[p2:p1, 0].real, alpha=0.1, color='red')
        if rev:
            plt.gca().invert_xaxis()

    def plotsppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-'):
        """
        This function plots a set of 1D spectrum

        :param sig: the signal to plot
        :param ppm1: the lower bound of the ppm range to plot
        :param ppm2: the ppm value of the right edge of the plot
        :param rev: reverse the x-axis
        :param linewidth: the width of the line
        :param linestyle: '-' is a solid line, '--' is a dashed line, '-.' is a dash-dot line, ':' is a dotted line,
        defaults to - (optional)
        """
        p1 = int(self.ppm2p(ppm1, len(sig)))
        p2 = int(self.ppm2p(ppm2, len(sig)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        sig = np.squeeze(sig)
        g = plt.plot(np.flip(x), sig[p2:p1, :].real, linewidth=linewidth, linestyle=linestyle)
        if rev:
            plt.gca().invert_xaxis()

    def normalize(self,inp):
        """
        It takes the absolute value of the input, divides it by the maximum absolute value of the input, and then multiplies
        it by the exponential of the angle of the input

        :param inp: The input signal
        :return: The magnitude of the input divided by the maximum magnitude of the input, multiplied by the complex
        exponential of the angle of the input.
        """
        return (np.abs(inp) / np.abs(inp).max(axis=0)) * np.exp(np.angle(inp) * 1j)

    def plotppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-'):
        """
        This function plots a signal in the frequency domain, given the start and end ppm values, and the signal

        :param sig: the signal you want to plot
        :param ppm1: The starting point of the plot in ppm
        :param ppm2: The ppm value of the right side of the plot
        :param rev: True or False, whether to reverse the x-axis
        :param linewidth: The width of the line
        :param linestyle: '-' for solid line, '--' for dashed line, '-.' for dash-dot line, ':' for dotted line, defaults to
        - (optional)
        :return: The plot is being returned.
        """
        p1 = int(self.ppm2p(ppm1, len(sig)))
        p2 = int(self.ppm2p(ppm2, len(sig)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        sig = np.squeeze(sig)
        df = pd.DataFrame({'Real Signal (a.u.)': sig[p2:p1].real})
        df['Frequency(ppm)'] = np.flip(x)
        g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle)
        if rev:
            plt.gca().invert_xaxis()
        return g
        # gca = plt.plot(x,sig[p2:p1,0],linewidth=linewidth, linestyle=linestyle)

    def plot_basis2(self, basisset, ampl):
        """
        It plots the basis set, which is a matrix of column vectors, each of which is a spectrum.

        The first argument is the basis set, the second is the amplitude of the basis set.

        The function plots the basis set by plotting each column vector as a spectrum.

        The function also shifts the spectra by 2000 ppm, so that they don't overlap.

        The function also adds a legend to the plot, which is the name of the metabolite.

        The function also saves the plot as a .png file.

        The function also displays the plot.


        :param basisset: the basis set to be plotted
        :param ampl: the amplitude of the basis set
        """
        for i in range(0, len(basisset.T) - 1):
            self.plotppm(-2000 * i + fft.fftshift(fft.fft(ampl * basisset[:, i])), 0, 5, False)
        self.plotppm(-2000 * (i + 1) + fft.fftshift(fft.fft(basisset[:, i + 1])), 0, 5, True)
        plt.legend(self.met_name)
        self.savefig(False,"Basis" + str(ampl))
        plt.show()

    def plot_basis(self, ampl, fr, damp, ph, ):
        """
        This function plots the basis set for each metabolite in the basis set.

        :param ampl: the amplitude of the basis function
        :param fr: frequency of the basis set
        :param damp: damping factor
        :param ph: phase
        """
        for i in range(0, len(self.basisset.T)):
            ax = self.plotppm(-2 * (i+2) + fft.fftshift(fft.fft(ampl[0, i] * self.basisset[:, i] * np.exp(-1*damp*self.t.T)*
                                                  np.exp(-2 * np.pi * fr[0, 0] * self.t.T))).T, 0, 5, False)
            sns.despine(left=True,right=True,top=True)
            plt.text(.1, -2 * (i+2), self.met_name[i],fontsize=8)
            ax.tick_params(left=False)
            ax.set(yticklabels=[])

    def Lornz(self, ampl, f, d, ph, Crfr, Crd):
        """
        The function takes in the amplitude, frequency, delay, phase, and the change in frequency and delay, and returns the
        lorentzian lineshape

        :param ampl: Amplitude of the signal
        :param f: frequency
        :param d: distance to the source
        :param ph: phase
        :param Crfr: Carrier frequency
        :param Crd: The distance from the source to the receiver
        :return: the complex exponential function with the given parameters.
        """
        return ampl * np.multiply(np.multiply(np.exp(ph * 1j),
                                                    np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                     np.exp(-1*(d + Crd) * self.t.T))
    def Gauss(self, ampl, f, d, ph, Crfr, Crd):
        """
        The function takes in the amplitude, frequency, delay, phase, and the change in frequency and delay, and returns the
        guassian lineshape

        :param ampl: Amplitude of the Gaussian
        :param f: frequency
        :param d: damping factor
        :param ph: phase
        :param Crfr: Carrier frequency
        :param Crd: The decay rate of the signal
        :return: the Gaussian waveform.
        """
        return ampl * np.multiply(np.multiply(np.exp(ph * 1j),
                                                    np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                     np.exp(-1*(d + Crd) * self.t.T * self.t.T))

    def data_proc(self):
        """
        It takes a dataset, performs a water removal, normalizes it, and then performs a Fourier transform on it
        :return: the training and test data.
        """
        if self.wr[0] == True:
            self.dataset = watrem.init(self.dataset[:,:],self.t_step, self.wr[1])
            with open(self.data_dir_ny, 'wb') as f:
                np.save(f, self.dataset, allow_pickle=True)
        else:
            if self.data_dir_ny is not None:
                with open(self.data_dir_ny, 'rb') as f:
                    self.dataset = np.load(f)
        if self.data_conj == True:
            y = np.conj(self.dataset)
        else:
            y = self.dataset
        y = self.normalize(y)
        y_f = fft.fftshift(fft.fft(y, axis=0), axes=0)
        if self.quality_filt[0] == True:
            cond = np.max(np.real(y_f[self.quality_filt[1]:self.quality_filt[2], :]), axis=0)
            idx = np.where((cond > 3) & (cond < 10))[0]
            y = y[0:2 * self.truncSigLen, idx]
        Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y[:, -self.test_nos:], 0), np.size(y[:, -self.test_nos:], 1),
                                     self.t_step * 1000, 0, 0, self.trnfreq*1e6), y[:, -self.test_nos:],
                    self.saving_dir + self.test_name)
        return y[:, 0:-self.test_nos],y[:, -self.test_nos:]


    def data_prep(self):
        """
        It takes in the simulation parameters and generates the signals, normalizes them, and then converts them as tensors.
        """
        if self.sim_params is not None:
            y, f, p, w_idx,l_idx = self.getSignals(self.sim_params[0],self.sim_params[1],self.sim_params[2]
                                            ,self.sim_params[3],self.sim_params[4],self.sim_params[5],
                                            self.sim_params[6],self.sim_params[7])
            y = self.normalize(y)
            if "dSR" in self.type:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(np.abs(y.T))
                mean_cl = np.mean(kmeans.cluster_centers_[:, 0:10], axis=1)
                y_non_cont = y[:, kmeans.labels_ == (np.argmin(mean_cl))]
                y_non_cont_f = fft.fftshift(fft.fft(y_non_cont, axis=0), axes=0)
                # ref_idx = np.argmax(self.cal_snrf())
                # (np.std(data_f.real[:endpoints, :], axis=0))
                y_non_cont_f_trunc = y_non_cont_f[self.ppm2p(4, self.sigLen):self.ppm2p(2.5, self.sigLen), :]
                snr_cr = np.max(np.abs(y_non_cont_f_trunc), axis=0) / (
                            np.std(y_non_cont.real[-256:, :], axis=0) * np.sqrt(self.sigLen))
                ref_idx = np.argmax(snr_cr)
                # cc = np.zeros(24000)
                # cc[w_idx] = 1
                # cc[l_idx] = 1
                # zz = kmeans.labels_ - cc
                # acc = np.size(np.where(zz == -1)) / 24000 + np.size(np.where(zz == 1)) / 24000
                self.dSR_refsignal = y_non_cont[:,ref_idx]
                self.plotppm(fft.fftshift(fft.fft(self.dSR_refsignal)),0,5,True)
                plt.title("Refrence Signal for dSR")
                plt.show()
                freq_indx = f[kmeans.labels_ == (np.argmin(mean_cl))][ref_idx]
                ph_indx = p[kmeans.labels_ == (np.argmin(mean_cl))][ref_idx]
                with open(os.path.join(self.saving_dir, 'dSR_refsignal' + '.txt'),'w') as fi:
                    fi.write("freq_offset:{}, and phase_offset:{}".format(freq_indx, ph_indx*180/np.pi))
                np.save(os.path.join(self.saving_dir, 'shifts_refsignal' + '.npy'),[freq_indx, ph_indx])
                np.save(os.path.join(self.saving_dir, 'dSR_refsignal' + '.npy'), self.dSR_refsignal)
            # self.refsignal = y[:,0]
        else:
            y_train, y_test = self.data_proc()
        if self.aug_params is not None:
            y, _, _, _, _ = self.data_aug(y_train[0:self.sigLen,:])
        # y_f = fft.fftshift(fft.fft(y, axis=0),axes=0)

        sns.set(style="white", palette="muted", color_codes=True)
        plt.hist(self.cal_snr(y,128), color="m")
        sns.despine()
        self.savefig(False,"snr_of_trainin_samples")
        plt.show()
        self.plotppm(fft.fftshift(fft.fft((y[:, 0]), axis=0),axes=0), 0, 5, True, linewidth=1, linestyle='-')
        self.savefig(False,"sample_sig")
        plt.show()
        plt.plot(range(0, 512), y[0:512, 0])
        plt.show()
        self.numOfSample = np.shape(y)[1];
        y_norm = y
        del y
        self.to_tensor(y_norm,np.asarray([f,p]))


    def data_aug(self,y):
        """
        It takes in a dataset, and returns a randomly augmented version of that dataset

        :param y: the image to be augmented
        :return: The augmented image is being returned.
        """
        return self.get_augment(y, self.aug_params[0], self.aug_params[1], self.aug_params[2], self.aug_params[3], self.aug_params[4], self.aug_params[5])


    def to_tensor(self,y_norm,labels):
        """
        This function takes in the normalized signal and the labels and converts them into tensors

        :param y_norm: the normalized signal
        :param labels: the labels of the data
        """
        labels = torch.from_numpy(labels.T)
        y_trun = y_norm[0:self.truncSigLen, :].astype('complex64')
        self.y_trun = torch.from_numpy(y_trun[:, 0:self.numOfSample].T)
        my_dataset = TensorDataset(self.y_trun,labels)
        self.train, self.val = random_split(my_dataset, [int((self.numOfSample) * self.tr_prc), self.numOfSample - int((self.numOfSample) * self.tr_prc)])
        self.my_dataloader = DataLoader(my_dataset, shuffle=True)


    def inputSig(self,x):
        """
        If the input shape is 2 channels, then return a tensor with the real and imaginary parts of the input signal.

        If the input shape is real, then return the real part of the input signal.

        :param x: the input signal
        :return: The real part of the signal.
        """
        if self.in_shape == '2chan':
            return torch.cat((torch.unsqueeze(x[:, 0:self.truncSigLen].real, 1), torch.unsqueeze(x[:, 0:self.truncSigLen].imag, 1)),1)
        if self.in_shape == 'real':
            return x[:, 0:self.truncSigLen].real

    # %%
    def testmodel(self,model, x):
        """
        The function takes in a model, and a tensor x, and returns the output of the model when x is passed through it

        :param model: the model you want to test
        :param x: the input data
        :return: The output of the model.
        """
        model.eval()
        with torch.no_grad():
            temp = model.forward(x)
        return temp

    def getSignals(self,ns,f_band,ph_band,ampl_band,d_band, nauis, num, noiseLevel):
        """
        It generates a signal with a given frequency, phase, amplitude, and decay, and adds noise to it

        :param ns: number of signals
        :param f_band: frequency band
        :param ph_band: phase band
        :param ampl_band: the amplitude of the signal
        :param d_band: the decay rate of the signal
        :param nauis: if True, then the signals will have noise added to them
        :param num: number of signals to generate
        :param noiseLevel: the level of noise to add to the signal
        :return: the signal, the shift, the phase, the white noise index, and the lipd noise index.
        """
        shift = f_band * np.random.rand(ns) - (f_band/2)
        freq = -2 * math.pi * (shift) * self.t
        ph = ph_band * np.random.rand(ns) * math.pi - ((ph_band/2) * math.pi)
        ampl = ampl_band * np.random.normal(1, 0.1, ns)
        d = d_band*np.random.normal(2, 0.1, ns)
        y =self.normalize(self.refsignal)
        y = ampl * y

        w_idx = []
        l_idx = []
        # TODO: should be converted to soft-code
        # Adding noise to the signal.
        # Checking if the variable nauis is true.
        if nauis == True:
            #
            w_idx = (np.random.rand(num) * ns).astype(int)
            l_idx = (np.random.rand(num) * ns).astype(int)
            nauisigW = -self.t*np.random.normal(1, 0.1, num) * np.exp(
                -2 * np.pi * 1 * np.random.normal(0, 0.02, num) * self.t * 1j) * np.exp(
                -100 * np.random.normal(1, 0.1, num) * (self.t))
            y[:, w_idx] = y[:, w_idx] + 20*nauisigW[:, 0:num]
            # elif nauis == 'L':
            #     print('lipd cont')
            nauisigL = -np.random.normal(1, 0.1, num) * np.exp(
                -2 * np.pi * 1400 * np.random.normal(1, 0.02, num) * self.t * 1j) * np.exp(
                -300 * np.random.normal(1, 0.1, num) * (self.t)) * np.exp(
                -1 * np.random.normal(0.5, 0, num) * np.pi * 1j)
            y[:, l_idx] = y[:, l_idx] + 1*(np.conj(nauisigL[:, 0:num]))
        y = np.multiply(y * np.exp(ph * 1j), np.exp(freq * 1j))
        y = np.multiply(y, np.exp(-d * self.t))
        noise = np.random.normal(0, noiseLevel, (ns, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (ns, self.sigLen))
        y = y + noise.T
        return y, shift, ph, w_idx,l_idx

    # %%
    # adding random phases and frequencies\
    def getSignal(self,n, f, ph, ampl, d, nauis, num, noiseLevel):
        """
        It generates a signal with a given number of frequencies, phases, amplitudes, and damping factors, and adds noise to
        it

        :param n: number of signals
        :param f: frequency of the signal
        :param ph: phase shift
        :param ampl: amplitude of the signal
        :param d: the decay rate of the signal
        :param nauis: whether to add noise or not
        :param num: number of signals
        :param noiseLevel: the level of noise to add to the signal
        :return: the signal, the signal without noise, the shift, the phase, the indices of the wideband and the indices of
        the longband noise.
        """
        shift = f * np.ones(n)
        freq = -2 * math.pi * (shift) * self.t
        ph = ph * np.ones(n)
        ampl = ampl * np.ones(n)
        d = d * np.ones(n)
        y = self.normalize(self.refsignal)
        y = ampl * y
        w_idx = (np.random.rand(num) * n).astype(int)
        l_idx = (np.random.rand(num) * n).astype(int)
        # TODO make soft
        if nauis == True:
            if np.random.rand() > 0.5:
                nauisigW = -self.t * np.random.normal(1, 0.1, num) * np.exp(
                    -2 * np.pi * 1 * np.random.normal(0, 0.02, num) * self.t * 1j) * np.exp(
                    -100 * np.random.normal(1, 0.1, num) * (self.t))
                y[:, w_idx] = y[:, w_idx] + 20 * nauisigW[:, 0:num]
            else:
                nauisigL = -np.random.normal(1, 0.1, num) * np.exp(
                    -2 * np.pi * 1400 * np.random.normal(1, 0.02, num) * self.t * 1j) * np.exp(
                    -300 * np.random.normal(1, 0.1, num) * (self.t)) * np.exp(
                    -1 * np.random.normal(0.5, 0, num) * np.pi * 1j)
                y[:, l_idx] = y[:, l_idx] + 1 * (np.conj(nauisigL[:, 0:num]))
        y = np.multiply(y * np.exp(ph * 1j), np.exp(freq * 1j))
        y = np.multiply(y, np.exp(-d * self.t ))
        noise = np.random.normal(0, noiseLevel, ((self.sigLen), n)) + 1j * np.random.normal(0, noiseLevel,
                                                                                          (self.sigLen, n))
        y_wn = y
        y = y_wn + noise
        return y, y_wn, shift, ph, w_idx, l_idx


    def testAsig(self,ph, f, n, nuis, nl):
        """
        It takes a signal, encodes it, and then decodes it

        :param ph: phase of the signal
        :param f: frequency of the signal
        :param n: number of signals to generate
        :param nuis: the noise level
        :param nl: number of layers
        :return: the latent space representation of the signal, the original signal, and the reconstructed signal.
        """
        sns.set_style('white')
        yxx,y_xx_wn, shift_t, ph_t, w_idx, l_idx = self.getSignal(n, f, ph, 1, 1, nuis, 1, nl)
        yxx = self.normalize(yxx)
        id = "testasig/" + str(nuis) + str(shift_t) + "_" + str(ph_t) + "_nl_" + str(nl) + "_n_" + str(n)
        Path(self.saving_dir + "testasig/").mkdir(parents=True, exist_ok=True)
        if n == 1:
            self.plotppm(np.fft.fftshift(np.fft.fft((yxx[0:self.truncSigLen]), axis=0)), 0, 5, False, linewidth=0.3,
                    linestyle='-')
        # plt.show()
        yxx = yxx.astype('complex64')
        yxx_t = torch.from_numpy(yxx.T)
        c = self.testmodel(self.autoencoders[0].to('cpu').encoder, self.inputSig(yxx_t))
        c = c.detach().numpy()
        print(c)
        print([ph_t, shift_t])
        rec_signal,_,_ = self.autoencoders[0].to('cuda')(torch.unsqueeze(yxx_t[0, 0:self.truncSigLen].cuda(), 0))
        if n == 1:
            self.plotppm(np.fft.fftshift(np.fft.fft((rec_signal.cpu().detach().numpy()[0, 0:self.truncSigLen])).T), 0, 5,
                    True, linewidth=1, linestyle='--')
            sns.despine()
            self.savefig(False,id + "_testAsig")
            plt.show()
        return c, yxx, rec_signal


    def test_time(self,n, ph_band, f_band, load, nauis, num, nl):
        """
        It takes a signal, normalizes it, and then runs it through the encoder of the  autoencoder to recorde the spending time

        :param n: number of signals
        :param ph_band: phase band
        :param f_band: frequency band of the signal
        :param load: whether to load the data from the file or generate it
        :param nauis: number of atoms in the dictionary
        :param num: number of signals
        :param nl: noise level
        """
        cmap = 'Blues'
        id = "time_" + str(nauis) + str(f_band) + "_" + str(ph_band) + "_nl_" + str(nl) + "_pr_" + str(num) + "_" + str(n)
        if load:
            y_test = np.load(os.path.join(self.test_data_root, id + 'y_t' + '.npy'))
            w_idx = np.load(os.path.join(self.test_data_root, id + 'w_idx' + '.npy'))
            shift_t = np.load(os.path.join(self.test_data_root, id + 'shift_t' + '.npy'))
            l_idx = np.load(os.path.join(self.test_data_root, id + 'l_idx' + '.npy'))
            ph_t = np.load(os.path.join(self.test_data_root, id + 'ph_t' + '.npy'))
            snrs = np.load(os.path.join(self.test_data_root, id + 'snrs_t' + '_.npy'))
        else:
            # min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond
            y_test, shift_t, ph_t, w_idx, l_idx = self.getSignals(n, f_band, ph_band, 1, 1, nauis, num, nl)
            np.save(os.path.join(self.test_data_root, id + 'y_t' + '.npy'), y_test)
            np.save(os.path.join(self.test_data_root, id + 'w_idx' + '.npy'), w_idx)
            np.save(os.path.join(self.test_data_root, id + 'shift_t' + '.npy'), shift_t)
            np.save(os.path.join(self.test_data_root, id + 'l_idx' + '.npy'), l_idx)
            np.save(os.path.join(self.test_data_root, id + 'ph_t' + '.npy'), ph_t)
            snrs = self.cal_snr(y_test,128)
            np.save(os.path.join(self.test_data_root, id + 'snrs_t' + '_.npy'), snrs)
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), y_test,
                        self.test_data_root + id + 'testDBjmrui.txt')
            sio.savemat(self.test_data_root + id + "testDB.mat",
                        {'y_test': y_test, 'f': shift_t, 'ph': ph_t})

        id = "test/" + id + "/"
        sns.set_palette('deep')
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        sns.set_style(style="whitegrid")


        # normalize
        y_test = self.normalize(y_test)
        yxx_t = torch.from_numpy(y_test.T.astype('complex64'))
        self.tic()
        c = self.testmodel(self.autoencoders[0].encoder, self.inputSig(yxx_t.cuda()))
        self.toc(id + str(n))


    # %%  test with n artificial signals
    def test(self,n, ph_band, f_band, load, nauis, num, nl):
        """
        It loads the test data, runs the model on it, and then plots the results

        :param n: number of signals
        :param ph_band: the phase band to use for the test data
        :param f_band: the frequency band of the signal
        :param load: whether to load the data from the disk or generate it
        :param nauis: number of AUIS
        :param num: number of signals
        :param nl: noise level
        :return: The corrected signal.
        """
        cmap = 'Blues'
        id = str(nauis) + str(f_band) + "_" + str(ph_band) + "_nl_" + str(nl) + "_pr_" + str(num) + "_" + str(n)
        if load:
            y_test = np.load(os.path.join(self.test_data_root, id + 'y_t' + '.npy'))
            w_idx = np.load(os.path.join(self.test_data_root, id + 'w_idx' + '.npy'))
            shift_t = np.load(os.path.join(self.test_data_root, id + 'shift_t' + '.npy'))
            l_idx = np.load(os.path.join(self.test_data_root, id + 'l_idx' + '.npy'))
            ph_t = np.load(os.path.join(self.test_data_root, id + 'ph_t' + '.npy'))
            snrs = np.load(os.path.join(self.test_data_root, id + 'snrs_t' + '_.npy'))
        else:
            # min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond
            y_test, shift_t, ph_t, w_idx, l_idx = self.getSignals(n, f_band, ph_band, 1, 1, nauis, num, nl)
            np.save(os.path.join(self.test_data_root, id + 'y_t' + '.npy'), y_test)
            np.save(os.path.join(self.test_data_root, id + 'w_idx' + '.npy'), w_idx)
            np.save(os.path.join(self.test_data_root, id + 'shift_t' + '.npy'), shift_t)
            np.save(os.path.join(self.test_data_root, id + 'l_idx' + '.npy'), l_idx)
            np.save(os.path.join(self.test_data_root, id + 'ph_t' + '.npy'), ph_t)
            snrs = self.cal_snr(y_test,128)
            np.save(os.path.join(self.test_data_root, id + 'snrs_t' + '_.npy'), snrs)
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), y_test,
                        self.test_data_root + id + 'testDBjmrui.txt')
            sio.savemat(self.test_data_root + id + "testDB.mat",
                        {'y_test': y_test, 'f': shift_t, 'ph': ph_t})

        id = "test/" + id + "/"
        sns.set_palette('deep')
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        sns.set_style(style="whitegrid")


        # normalize
        y_test = self.normalize(y_test)
        yxx_t = torch.from_numpy(y_test.T.astype('complex64'))
        self.tic()
        c = self.testmodel(self.autoencoders[0].encoder, self.inputSig(yxx_t.cuda()))
        self.toc(id + str(n))
        c = c.cpu().detach().numpy()

        if 'dSR' in self.type:
            ref = np.load(os.path.join(self.saving_dir, 'shifts_refsignal' + '.npy'))
            c = c + [ref[1],ref[0]]
        else:
            c = -1 * c


        mean_f = np.mean((c[:, 1]) - shift_t)
        mean_ph = np.mean((c[:, 0]) - ph_t) * 180 / np.pi
        std_f = np.std((c[:, 1]) - shift_t)
        std_ph = np.std((c[:, 0]) - ph_t) * 180 / np.pi
        print(mean_f)
        print(mean_ph)
        print(std_f)
        print(std_ph)
        np.savetxt(self.saving_dir + id + 'mean_std.txt', (mean_f, mean_ph, std_f, std_ph))
        monttxt = open(self.saving_dir + id + 'error_freq_line' + str(f_band) + "_" + str(ph_band) + ".txt", 'w')
        monttxt.write(str(stats.linregress(shift_t, (c[:, 1]))))

        df = pd.DataFrame(data=shift_t, columns=['True Frequency'])
        df['True Phase'] = (180 / np.pi) * ph_t
        df['Predicted Frequency'] = c[:, 1]
        df['Predicted Phase'] = (180 / np.pi) * c[:, 0]
        df['Frequency Error'] = df['Predicted Frequency'] - df['True Frequency']
        df['Phase Error'] = df['Predicted Phase'] - df['True Phase']
        idx = np.zeros((n, 1));
        if nauis:
            idx[w_idx] = idx[w_idx] + 1
            idx[l_idx] = idx[l_idx] + 2
        df['classes'] = idx
        df['classes'][df['classes'] == 3] = 'UW & LC'
        df['classes'][df['classes'] == 2] = 'LC'
        df['classes'][df['classes'] == 1] = 'UW'
        df['classes'][df['classes'] == 0] = 'Free'
        if max(idx) == 3: markers = ["o", "x", "1", "*"]
        if max(idx) == 2: markers = ["o", "x", "1"]
        if max(idx) == 1: markers = ["o", "x"]
        if max(idx) == 0: markers = ["o"]
        df['SNR'] = snrs
        sns.set_style('whitegrid')
        sns.distplot(df['SNR'], color="m")
        sns.despine()
        self.savefig(False,"test_snr_hist")
        plt.show()
        sns.lmplot(x="True Frequency", y="Frequency Error", data=df, fit_reg=False, hue='classes', legend=True, markers=markers)
        self.savefig(False,id + 'error_freq')
        plt.show()
        sns.lmplot(x="True Phase", y="Phase Error", data=df, fit_reg=False, hue='classes', legend=True, markers=markers)
        self.savefig(False,id + 'error_phase')
        plt.show()

        g = sns.jointplot(x="True Frequency", y="Frequency Error",data=df)
        self.savefig(False,id + 'error_freq2')
        plt.show()

        monttxt = open(self.saving_dir + id + 'error_ph_line' + ".txt", 'w')
        monttxt.write(str(stats.linregress(ph_t, (c[:, 0]))))
        g = sns.jointplot(x="True Phase", y="Phase Error",data=df)
        self.savefig(False,id + 'error_phase2')
        plt.show()
        # qfc = np.abs(c - [ph_t, shift_t]) / (np.abs(c) + np.abs([ph_t, shift_t]))
        q = (np.abs(c[:, 1] - shift_t) / (np.max((np.abs(c[:, 1] - shift_t))))) + (
                np.abs(c[:, 0] - ph_t) / (np.max(np.abs(c[:, 0] - ph_t))))
        # q = np.abs(np.abs(c[:, 1] - shift_t)/shift_t) + np.abs(np.abs((c[:, 0]) - ph_t)/ph_t)
        plt.scatter(shift_t, 180 / np.pi * ph_t, c=q / 2, cmap='Spectral')
        cmap = 'Reds'
        plt.set_cmap(cmap)
        plt.colorbar()
        self.savefig(False,id + 'joint')
        plt.show()

        q = np.abs(c[:, 1] - shift_t)
        # q = np.abs(np.abs(c[:, 1] - shift_t)/shift_t) + np.abs(np.abs((c[:, 0]) - ph_t)/ph_t)
        plt.scatter(shift_t, 180 / np.pi * ph_t, c=q, cmap='Spectral')
        plt.set_cmap(cmap)
        plt.colorbar()
        self.savefig(False,id + 'joint_free')
        plt.show()

        q = 180 / np.pi * np.abs(c[:, 0] - (ph_t))
        plt.scatter(shift_t, 180 / np.pi * ph_t, c=q, cmap='Spectral')
        plt.set_cmap(cmap)
        plt.colorbar()
        self.savefig(False,id + 'joint_ph')
        plt.show()
        l = int(n/12)
        sns.set_style('white')
        sns.set_palette('pastel')
        self.plotsppm(fft.fftshift(fft.fft((y_test[:, 0:l]), axis=0),axes=0),0,5,False,linewidth=0.3)
        self.plotppm(fft.fftshift(fft.fft((np.mean(y_test[:, 0:l],axis=1)), axis=0), axes=0), 0, 5, True, linewidth=1)
        sns.despine(left=False,right=False,top=False)
        self.savefig(False, id + "_" + "y_test")
        plt.show()
        rec_signal = (np.exp(-1 * c[:, 0] * 1j) * np.multiply(y_test[:, :], np.exp(
            2 * math.pi * (c[:, 1]) * self.t * 1j)))
        self.plotsppm(fft.fftshift(fft.fft((rec_signal[:, :]), axis=0), axes=0), 0, 5, False)
        self.plotppm(fft.fftshift(fft.fft((np.mean(rec_signal[:, :], axis=1)), axis=0), axes=0), 0, 5, True, linewidth=1)
        self.savefig(False,id + "_" + "y_test_after")
        plt.show()
        np.save(os.path.join(self.saving_dir, id + self.parameters['version'][0:-1]+ '_corrected' + '.npy'),rec_signal)
        df.to_csv(self.saving_dir+id+self.parameters['version'][0:-1]+"_rslt.csv")
        return rec_signal

    def monteCarlo(self,n, nuis, nl, f, ph,load):
        """
        It takes in a bunch of parameters, and then it either loads the data from a file or generates it,
        and then apply a Monte Calrlo analysis on
        the data, and finally it saves the result

        :param n: number of samples
        :param nuis: noise level
        :param nl: noise level
        :param f: frequency of the signal
        :param ph: phase of the signal
        :param load: whether to load the data from the file or generate it
        """
        id = str(n) + str(nuis) + "_" + str(nl) + "_f_" + str(f) + "_ph_" + str(np.round(ph))
        if load:
            yxx = np.load(os.path.join(self.test_data_root, id + 'y_t_mc' + '.npy'))
            c = np.load(os.path.join(self.test_data_root, self.parameters['version']+id + 'c_mc' + '.npy'))
            snrs = np.load(os.path.join(self.test_data_root, id + 'snrs_mc' + '.npy'))
        else:
            # min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond
            c, yxx, _ = self.testAsig(ph, f, n, nuis, nl)
            np.save(os.path.join(self.test_data_root, self.parameters['version']+id + 'c_mc' + '.npy'), c)
            np.save(os.path.join(self.test_data_root, id + 'y_t_mc' + '.npy'), yxx)
            snrs = self.cal_snr(yxx,128)
            np.save(os.path.join(self.test_data_root, id + 'snrs_mc' + '.npy'), snrs)
            Jmrui.write(Jmrui.makeHeader("mc", np.size(yxx, 0), np.size(yxx, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), yxx,
                        self.test_data_root + id + 'mc.txt')

        id = "mc/" + id + "/"

        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        sns.set(style="white", palette="muted", color_codes=True)

        c = -1*c

        rslt_hlsvd = self.cal_snr_lw(self.refsignal)
        # Crfr = self.trnfreq * (4.7 - self.comp_freq)
        pre_ph = ((c[:, 0]) * 180 / np.pi)
        pre_f = (c[:, 1])
        # pre_ph = ((c[:, 0]) * 180 / np.pi)+rslt_hlsvd[5][0]
        # pre_f = (c[:, 1]) + (rslt_hlsvd[2][0]*1000-(Crfr))
        df = pd.DataFrame()
        df['Predicted Frequency'] = pre_f
        df['Predicted Phase'] = pre_ph
        df['SNR'] = snrs
        monttxt = open(self.saving_dir + id + "menan.txt", 'w')
        monttxt.write(" mean_f:" + str(np.mean(pre_f)))
        monttxt.write(" mean_ph:" + str(np.mean(pre_ph)))
        monttxt.write(" std_f:" + str(np.std(pre_f)))
        monttxt.write(" std_ph:" + str(np.std(pre_ph)))
        monttxt.close()
        sns.jointplot(x=pre_f, y=pre_ph, kind='hist')
        self.savefig(False,id + "bivariantehist")
        plt.show()
        sns.jointplot(x=pre_f, y=pre_ph, kind='hist', cbar=True)
        self.savefig(False,id + "bivariantehistWithbar")
        plt.show()
        df.to_csv(self.saving_dir+id+self.parameters['version'][:-1]+"_mc_rslt.csv")


    def watrem(self,data, dt, n):
        """
        The function takes in a time series, the sampling interval, and the number of singular values to return. It then
        performs the HLSVD on the time series, and returns the singular values, frequencies, damping factors, amplitudes,
        and phases

        :param data: the data to be analyzed
        :param dt: time step in seconds
        :param n: number of singular values to return
        :return: the FID and the result.
        """
        npts = len(data)
        dwell = dt / 0.001
        nsv_sought = n
        result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
        nsv_found, singvals, freq, damp, ampl, phas = result
        Crfr = self.trnfreq * (4.7 - 3.027)
        idx = np.where((result[2] < (0.001 * (Crfr + 50))) & (result[2] > (0.001 * (Crfr - 50))))
        result = (len(idx), result[1], result[2][idx], result[3][idx], result[4][idx], result[5][idx])
        fid = hlsvdpro.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)
        return fid, result


    def cal_snr_lw(self,av):
        """
        It takes the average of the FID, removes the water peak, and then calculates the SNR of the resulting spectrum.


        :param av: the average spectrum
        :return: The SNR of the signal.
        """
        av_f = fft.fftshift(fft.fft((av),axis=0))
        self.plotppm(av_f, 0, 5, False)
        lsr, rslt = self.watrem(av, self.t_step, 5)
        lsr = fft.fftshift(fft.fft(((lsr))))
        # plotppm(lsr, 0,5,False)
        # av = lsr
        # x0 = (0, 0, 0, 0)
        # res = least_squares(cf, x0, bounds=([0,-10,0,-np.inf],[np.inf,10,100,np.inf]),method='dogbox')
        # lsr = fft.fftshift(fft.fft(np.conj(lsr.T)))
        # lsr = fft.fftshift(fft.fft(np.conj(Lornz(res.x))))
        self.plotppm(lsr, 0, 5, True)
        # plt.title('Linewidth: '+ str(-1000/(np.pi*res.x[2])) + "Hz" + "SNR: " + str(snr))

        self.savefig(False, "snr*lw")
        plt.show()
        return rslt

    def erroVsnoise(self, n, nuis, nl, f, ph,load):
        """
        This function takes in the number of signals to be generated, the noise level, the frequency and phase of the
        signal, and a boolean value to load the data or not. It then generates the signals, calculates the SNR, and then
        runs the signals through the trained model to get the predicted frequency and phase. It then plots the error vs SNR.

        :param n: number of samples
        :param nuis: noise
        :param nl: noise level
        :param f: frequency of the signal
        :param ph: phase of the signal
        :param load: if you want to load the data from the file
        """
        id = str(n) + str(nuis) + "_" + str(nl) + "_f_" + str(f) + "_ph_" + str(np.round(ph))
        if load:
            yxx = np.load(os.path.join(self.test_data_root, id + 'errvsn_y' + '.npy'))
            # c = np.load(os.path.join(self.test_data_root, self.parameters['version']+id + 'errvsn_c' + '.npy'))
            # snrs = np.load(os.path.join(self.test_data_root, id + 'snrs_errvsn' + '.npy'))
            snrs = self.cal_snr(yxx,128)
        else:
            yxx = np.zeros((self.sigLen, n), dtype='complex128')
            c  =np.zeros((n,2))
            nl_local = np.linspace(((0*nl)/n),nl,n)
            for i in range(0, n):
                yxx_temp, _, _, _, _, _ = self.getSignal(1, f, ph, 1, 1, nuis, 1, nl_local[i])
                # c_temp, yxx_temp, _ = self.testAsig(ph, f, 1, nuis, nl_local[i])
                yxx[:,i] = np.squeeze(yxx_temp)
                # c[i]= c_temp[0,0:2]

            snrs = self.cal_snr(yxx,128)
            np.save(os.path.join(self.test_data_root, id + 'snrs_errvsn' + '.npy'), snrs)
            np.save(os.path.join(self.test_data_root, id + 'errvsn_y' + '.npy'), yxx)
            np.save(os.path.join(self.test_data_root,self.parameters['version']+ id + 'errvsn_c' + '.npy'), c)
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(yxx, 0), np.size(yxx, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), yxx,
                        self.test_data_root + id + 'errorvsn_y_t.txt')
            sio.savemat(os.path.join(self.test_data_root, id + 'snrs_errvsn' + '.mat'),{"y_test":yxx})
        y_test = self.normalize(yxx)
        yxx_t = torch.from_numpy(y_test.T.astype('complex64'))
        c = self.testmodel(self.autoencoders[0].to('cpu').encoder, self.inputSig(yxx_t))
        c = (c).detach().numpy()



        id = "eVSn/" + id + "/"
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)

        # rslt_hlsvd = self.cal_snr_lw(self.refsignal)
        # Crfr = self.trnfreq * (4.7 - self.comp_freq)

        pre_ph = ((c[:, 0]) * 180 / np.pi)
        pre_f = (c[:, 1])
        df = pd.DataFrame()
        df['SNR'] = snrs
        df['Predicted Frequency'] = pre_f
        df['Predicted Phase'] = pre_ph
        df['Frequency Error'] = np.abs(pre_f - f)
        df['Phase Error'] = np.abs(pre_ph - (ph* 180 / np.pi))
        g = sns.histplot(snrs, color="skyblue", kde=True)
        self.savefig(False,id + "snrs_errVsnoise")
        plt.show()
        sns.regplot(x='SNR', y='Frequency Error', data=df, order=2, ci=None, line_kws={"color": "darkblue"})
        plt.legend(["frquency"])
        self.savefig(False,id + "_frq_errorvssnrRslt")
        plt.show()
        sns.regplot(x='SNR', y='Phase Error',data=df, order=2, ci=None, line_kws={"color": "darkblue"})
        plt.legend(["phase"])
        self.savefig(False,id + "_phase_errorvssnrRslt")
        plt.show()
        df.to_csv(self.saving_dir+id+self.parameters['version'][:-1]+"_evn_rslt.csv")




    def dotrain(self):
        """
        The function takes in a list of hyperparameters and trains a list of autoencoders
        """

        if self.MM_plot == True:
            if 'param' in self.MM_type:
                mm = 0
                for idx in range(0, self.numOfMM):
                    x = np.conj(self.MM_model(self.MM_a[idx], 0, 0, 0, self.ppm2f(self.MM_f[idx]), self.MM_d[idx]))
                    mm += x
                    if idx == self.numOfMM - 1:
                        self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, True)
                    else:
                        self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, False)
                self.savefig(False,"MM")
                plt.show()
                self.mm = mm.T
        if self.basis_need_shift[0] == True:
            self.basisset = self.basisset * np.exp(2 * np.pi * self.ppm2f(self.basis_need_shift[1]) * 1j * self.t)
        if self.basis_plot == True:
            self.plot_basis2(self.basisset, 2)
        if self.tr is True:
            self.data_prep()
            autoencoders = []
            self.tic()
            for i in range(0,self.ens):
                pl.seed_everything(42)
                logger = TensorBoardLogger('tb-logs', name=self.loging_dir)
                if self.parameters['early_stop'][0]:
                    early_stopping = EarlyStopping('val_loss',patience=self.parameters['early_stop'][1])
                    trainer= pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[early_stopping])
                else:
                    trainer = pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger)
                logger.save()
                temp = Encoder_Model(self.depths[i],self)
                trainer.fit(temp.to('cuda:0'), DataLoader(self.train, batch_size=self.batchsize), DataLoader(self.val, batch_size=self.batchsize))
                autoencoders.append(temp)
                PATH = self.saving_dir + "model_"+ str(i) + ".pt"
                # Save
                torch.save(temp.state_dict(), PATH)
                del temp
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.memory_summary(device=None, abbreviated=False)
            self.toc("trining_time")

    def dotest(self):
        """
        This function is used to evaluate the performance of the trained model
        """
        print("evaluation")
        self.autoencoders = []
        for i in range(0, self.ens):
            device = torch.device('cuda:0')
            model = Encoder_Model(self.depths[i],self)
            PATH = self.saving_dir + "model_" + str(i) + ".pt"
            model.load_state_dict(torch.load(PATH, map_location=device))
            model.cuda()
            model.eval()
            self.autoencoders.append(model)
            # x = summary(model)
            # print(x)
        self.testAsig(0.25, 5, 1, False, 0.05)
        self.test(128, 1, 40, self.parameters['test_load'], False, 64, 0.05)
        self.test(128, 1, 40, self.parameters['test_load'], True, 64, 0.05)
        self.test(128, 1, 40, self.parameters['test_load'], True, 64, 0.01)
        self.monteCarlo(256, False, 0.05, 5, 0.25*np.pi,False)
        self.erroVsnoise(20, False, 0.075, 5, 0.25*np.pi,self.parameters['test_load'])

        self.test_time(10000, 1, 40, self.parameters['test_load'], True, 64, 0.05)


    def tuner(self):
        """
        We are using TuneBOHB as our search algorithm, HyperBandForBOHB as our scheduler, and CLIReporter as our progress
        reporter.

        We are using the following hyperparameters:

        - lr: learning rate
        - dp: dropout
        - batchsize: batch size
        - bn: batch normalization

        We are using the following metrics:

        - mean_accuracy
        - training_iteration

        We are using the following resources:

        - gpu: 1

        We are using the following configuration:

        - num_samples: 30
        - metric: mean_accuracy
        - mode: min
        - resources_per_trial: resources_per_trial
        - config: config
        - scheduler: bohb
        - progress_reporter: reporter
        - name: exp2a
        - search_alg: algo
        """
        config = {
            "lr": tune.loguniform(1e-6, 1e-1),
            "dp": tune.choice([3, 4, 5, 6]),
            "batchsize" : tune.choice([8, 32,128,512]),
            "bn": tune.choice([0, 1]),
        }
        # scheduler = ASHAScheduler(
        #     max_t=max_epoch,
        #     grace_period=1,
        #     reduction_factor=2)
        algo = TuneBOHB(metric="mean_accuracy", mode="min")
        bohb = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100)
        reporter = CLIReporter(
            parameter_columns=["lr","dp","batchsize","bn"],
            metric_columns=["mean_accuracy", "training_iteration"])

        train_fn_with_parameters = tune.with_parameters(tunermodel)
        resources_per_trial = {"gpu": 1}

        # analysis = tune.run(train_fn_with_parameters,
        #     num_samples=20,
        #     resources_per_trial=resources_per_trial,
        #     metric="mean_accuracy",
        #     mode="min",
        #     config=config,
        #     scheduler=scheduler,
        #     progress_reporter=reporter,
        #     name="tune_mnist_asha")
        analysis = tune.run(train_fn_with_parameters,
            num_samples=30,
            metric="mean_accuracy",
            mode="min",
            resources_per_trial=resources_per_trial,
            config=config,
            scheduler=bohb,
            progress_reporter=reporter,
            name="exp2a",
            search_alg=algo)
        print("Best hyperparameters found were: ", analysis.best_config)





