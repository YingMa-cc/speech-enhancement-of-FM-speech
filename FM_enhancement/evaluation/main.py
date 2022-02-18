import sys
import os
import torch.nn as nn
from scipy.io import loadmat, savemat

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import librosa
from net.net import Net
import torch
from utils.model_handle import resume_model
from utils.util import get_alpha,expandWindow
from config import *
import progressbar
from utils.stft_istft import STFT
import torchaudio.functional as taf
from pypesq import pesq
from pystoi import stoi

import collections
import contextlib
import numpy as np
from numpy import *
import wave
import os
# import webrtcvad
from scipy.io import wavfile, loadmat
import soundfile as sf
from load_data.SpeechDataLoad import FeatureCreator
from utils.label_set import LabelHelper
from torch.utils.data import DataLoader
from scipy import interpolate


import matplotlib.pyplot as plt
from PIL import Image

import pickle


def load_obj(root_dir, name):
    with open(root_dir + name, 'rb') as f:
        return pickle.load(f)


def test_module_with_real(path):
    NET = Net()
    NET = nn.DataParallel(NET)
    resume_model(NET, MODEL_STORE + '/model_10.pkl')
    NET.eval()
    NET.cuda(CUDA_ID[0])
    file = os.listdir(path)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])

    mix, fs = sf.read(path + file[1])
    mix_est = np.zeros((0,1))
    mix_origin = np.zeros((0))
    bar = progressbar.ProgressBar(0, int(len(mix) / 160000)+1)
    # bar = progressbar.ProgressBar(0,1)
    bar.start()
    for i in range(int(len(mix)/160000)+1):
    # for i in range(1):
        bar.update(i)
        if i == int(len(mix)/160000):
            mix_cut = mix[i * 160000:]
        else:
            mix_cut = mix[i*160000:(i+1)*160000]
        mix_spec = stft.transform(torch.Tensor(mix_cut[np.newaxis, :]).cuda(CUDA_ID[0]))

        est_spec = NET(mix_spec)

        # 恢复语音
        res = stft.inverse(est_spec)
        res = res.permute(1, 0).detach().cpu().numpy()
        mix_est = np.concatenate((mix_est, res),axis=0)
        mix_origin = np.concatenate((mix_origin, mix_cut[:len(res)]),axis=0)

    sf.write((RESULT_STORE + file[1][:-4] + 'est.wav'), mix_est/np.max(mix_est), 16000)
    sf.write((RESULT_STORE + file[1][:-4] + '.wav'), mix_origin, 16000)
    bar.finish()



def test_module(path):
    NET = Net()
    NET = nn.DataParallel(NET)
    resume_model(NET, MODEL_STORE + '/model_10.pkl')

    NET.eval()
    NET.cuda(CUDA_ID[0])
    file = os.listdir(path)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])
    mix_pesq = []
    est_pesq = []
    mix_stoi = []
    est_stoi = []
    bar = progressbar.ProgressBar(0, VALIDATION_DATA_NUM)
    bar.start()
    for i in range(VALIDATION_DATA_NUM):
        bar.update(i)
        data = load_obj(path, file[i])
        # data = loadmat(path+file[i])


        speech = np.array(data['speech'])

        mix = np.array(data['mix'])
        mix_t = torch.Tensor(mix[np.newaxis, :])
        mix_spec = stft.transform(mix_t.cuda(CUDA_ID[0]))

        est_spec = NET(mix_spec)

        # 恢复语音
        res1 = stft.inverse(est_spec)
        res1 = res1.permute(1, 0).detach().cpu().numpy()
        res = taf.istft(est_spec.permute(0,2,1,3).detach().cpu(), FILTER_LENGTH, HOP_LENGTH, FILTER_LENGTH, torch.hamming_window(FILTER_LENGTH), center=False)
        res = res.permute(1, 0)

        # est_mag = torch.sqrt(est_spec[:, :, :, 0] ** 2 + est_spec[:, :, :, 1] ** 2)
        # angle = taf.angle(mix_spec)
        # est_real = est_mag * torch.cos(angle)
        # est_imag = est_mag * torch.sin(angle)
        # est_spec = torch.stack([est_real, est_imag], 3)
        # res2 = stft.inverse(est_spec)
        # res2 = res2.permute(1, 0).detach().cpu().numpy()

        mix_pesq.append(pesq(speech,mix,16000))
        est_pesq.append(pesq(speech[:len(res)],res.squeeze(),16000))

        mix_stoi.append(stoi(speech,mix,16000))
        est_stoi.append(stoi(speech[:len(res)],res.squeeze(),16000))

        # sf.write((RESULT_STORE + file[i][:-4] + 'mix.wav'), mix[:len(res2)], 16000)
        # sf.write((RESULT_STORE + file[i][:-4] + 'clean.wav'), speech[:len(res2)], 16000)
        # sf.write((RESULT_STORE + file[i][:-4] + 'est.wav'), res2, 16000)

    bar.finish()
    nb_param_q = sum(p.numel() for p in NET.parameters() if p.requires_grad)
    print("Number of trainable parameters : " + str(nb_param_q))
    print("mix_pesq : " +str(np.mean(mix_pesq)))
    print("est_pesq: " +str(np.mean(est_pesq)))

    print("mix_stoi : " + str(np.mean(mix_stoi)))
    print("est_stoi: " + str(np.mean(est_stoi)))


def test(path):
    file = os.listdir(path)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])

    feature_creator = FeatureCreator()
    label_helper = LabelHelper().cuda(CUDA_ID[0])
    bar = progressbar.ProgressBar(0, 5)
    bar.start()
    for i in range(5):
        bar.update(i)
        data = load_obj(path, file[i])

        speech = np.array(data['speech'])
        mix = np.array(data['mix'])
        speech_spec = stft.transform(torch.Tensor(speech[np.newaxis, :]).cuda(CUDA_ID[0]))
        mix_spec = stft.transform(torch.Tensor(mix[np.newaxis, :]).cuda(CUDA_ID[0]))
        mix_mag = stft.spec_transform(mix_spec)

        label = label_helper(speech_spec, mix_spec)

        # label_n = label.cpu().numpy()
        #
        # label_im = Image.fromarray(label_n*255)
        # label_im = label_im.convert('L')

        # plt.figure()
        # plt.imshow(label_im)
        # plt.show()
        est_mag = mix_mag * label

        # 恢复语音
        # angle = taf.angle(mix_spec)
        # est_real = est_mag * torch.cos(angle)
        # est_imag = est_mag * torch.sin(angle)
        # est_spec = torch.stack([est_real, est_imag], 3)
        # res = stft.inverse(est_spec)
        # res = res.permute(1, 0).detach().cpu().numpy()
        # sf.write((RESULT_STORE + 'label/'+file[i][:-4] + 'mix.wav'), mix[:len(res)], 16000)
        # sf.write((RESULT_STORE +'label/'+ file[i][:-4] + 'clean.wav'), speech[:len(res)], 16000)
        # sf.write((RESULT_STORE +'label/'+ file[i][:-4] + 'est.wav'), res, 16000)
    bar.finish()


if __name__ == '__main__':
    test_module(TEST_DATA_PATH)
    # test_module_with_real('/data02/maying/data/FM_real_data/')
    # test(TEST_DATA_PATH)
