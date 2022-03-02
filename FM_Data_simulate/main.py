import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy import fftpack
from matplotlib import pyplot as plt
from pylab import mpl
import wave
import math
# import pyaudio
from scipy import signal
from scipy.io import wavfile
import os
import random
import soundfile as sf
import progressbar
import pickle


def wgn(x, snr):
    signal_add_noise = np.zeros((0))
    for i in range(int(len(x) / 100 + 1)):
        if i == int(len(x) / 100):
            x_cut = x[i * 100:]
        else:
            x_cut = x[i * 100:(i + 1) * 100]
        if len(x_cut) == 0:
            break
        ps = np.sum(abs(x_cut) ** 2) / len(x_cut)
        pn = ps / (10 ** (snr / 10))
        noise = np.random.randn(len(x_cut)) * np.sqrt(pn)
        res = x_cut + noise
        signal_add_noise = np.concatenate((signal_add_noise, res), axis=0)
    return signal_add_noise

def load_obj(root_dir, name):
    with open(root_dir + name, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            print(f)
            return None


def save_obj(obj, name):
    with open(store_path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


path = '/data01/maying/data/FM_data/clean_testset_wav/'

store_path = '/data01/maying/data/FM_large_data_2.2/cv/ETU/'
files = os.listdir(path)
end = 1000
bar = progressbar.ProgressBar(0, end)
bar.start()
num = 0
# 信号幅度正态分布参数
u = 0  # 均值μ
sig = math.sqrt(1)  # 标准差δ
for z in range(1000):
    for i in range(len(files)):
        num = num + 1
        bar.update(num)
        data, Fs = sf.read(path + files[i])

        f1 = data[::3]

        n = len(f1)
        t = np.arange(0, n) / 16000

        N = random.randint(5, 10)  # 路径数随机

        t_deviation = 0.08  # 最大时延设置为5000ns
        tau = t_deviation * (np.abs(np.random.rand(N)))  # 随机生成每条路径时延
        # 根据正态分布依据时延计算出每条路径的信号幅度
        a = np.exp(-(tau / t_deviation - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
        for j in range(N):
            f_d = random.uniform(295, 305)  # 多普勒扩展

            shift = np.random.rand(1, 1) * 2 * f_d - f_d  # Doppler shifts
            # 信号乘幅度、多径及多普勒
            f2 = a[j] * ifft(fft(f1) * np.exp(-1j * 2 * math.pi * (np.arange(0, n) / n) * tau[j])) * np.exp(
                1j * 2 * math.pi * shift[0] * t)

            if j == 0:
                signal = f1 / math.sqrt(2 * math.pi)
                # signal = 0
            signal = signal + f2.real

        snr = random.uniform(0, 10)
        signal = wgn(signal, snr)

        data = {'speech': f1 / math.sqrt(2 * math.pi), 'mix': signal}

        # sf.write((store_path + files[i][:-4] + '_EPA_' + str(N) + '_' + str(snr) + '.wav'), signal, 16000)
        # sf.write((store_path + files[i][:-4] + '.wav'), data['speech'], 16000)
        save_obj(data, files[i][:-4] + '_ETU_' + str(N) + '_' + str(snr))
        # save_obj(data, files[i][:-4] + '_' + str(snr))
        if num == end:
            break
    if num == end:
        break

bar.finish()
