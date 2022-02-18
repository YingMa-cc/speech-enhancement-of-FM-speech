import os
import torch
import numpy as np
import librosa
import scipy
import scipy.signal as signal
import torch.nn as nn
from scipy.io import loadmat
from utils.util import get_alpha
from utils.stft_istft import STFT
import soundfile as sf
from utils.label_set import LabelHelper
from config import *
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt


def load_obj(root_dir, name):
    with open(root_dir + name, 'rb') as f:
        return pickle.load(f)


def normalization(data):
    """
    均值方差归一化
    """
    return (data - data.mean()) / torch.sqrt(data.var())


class Data(object):

    def __init__(self, index, s_path, n_path, n_index, n_start, n_end, alpha, snr):
        super(Data, self).__init__()  # ???
        self.index = index
        self.s_path = s_path
        self.n_path = n_path
        self.n_index = n_index
        self.n_start = n_start
        self.n_end = n_end
        self.alpha = alpha
        self.snr = snr


# class SpeechDataLoader(object):
#     def __init__(self, data_set, batch_size, is_shuffle=True, num_workers=0):
#         """
#         初始化一个系统的Dataloader,只重写其collate_fn方法
#         :param data_set:送入网络的data,dataset对象
#         :param batch_size:每次送入网络的data的数量，即语音数量
#         :param is_shuffle:是否打乱送入网络
#         :param num_workers:dataloader多线程工作数，一般为0
#         """
#         self.data_loader = DataLoader(dataset=data_set,
#                                       batch_size=batch_size,
#                                       shuffle=is_shuffle,
#                                       num_workers=num_workers,
#                                       collate_fn=collate,
#                                       pin_memory=True)
#
#     def get_data_loader(self):
#         """
#         获取dataloader
#         :return: dataloader对象
#         """
#         return self.data_loader


class SpeechDataset(Dataset):

    def __init__(self, root_dir):
        """
        初始化dataset：读入文件的list
        :param root_dir: 文件的根目录
        """
        # 初始化变量
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def __getitem__(self, item):
        """
        对于每个送入网络的数据进行处理
        PS：一般只对数据进行时域上的操作，其它诸如STFT的操作送入CUDA后执行
        :param item:送入网络数据的索引，一般是文件的索引
        :return:输入的数据
        """
        data = load_obj(self.root_dir, self.files[item])

        # 迭代输出需要的文件
        speech = np.array(data['speech'])
        mix = np.array(data['mix'])

        return torch.Tensor(speech), torch.Tensor(mix)

    def __len__(self):
        """
        返回总体数据的长度
        :return: 总体数据的长度
        """
        return len(self.files)


class BatchInfo(object):

    def __init__(self, speech, mix, frame):
        self.speech = speech
        self.mix = mix
        self.frame = frame


class FeatureCreator(nn.Module):

    def __init__(self):
        super(FeatureCreator, self).__init__()
        self.stft = STFT(FILTER_LENGTH, HOP_LENGTH).cuda(CUDA_ID[0])
        self.label_helper = LabelHelper().cuda(CUDA_ID[0])

    def forward(self, batch_info):
        """
        对数据进行短时傅里叶变换及相关操作，返回label及混合语音时频谱
        :param batch_info:相应数据及帧长信息
        :return:幅度谱，标签，帧长
        """

        speech_spec = self.stft.transform(batch_info.speech.transpose(1, 0).cuda(CUDA_ID[0]))
        mix_spec = self.stft.transform(batch_info.mix.transpose(1, 0).cuda(CUDA_ID[0]))
        speech_mag = self.stft.spec_transform(speech_spec)
        # speech_spec = torch.stft(batch_info.speech.transpose(1, 0), FILTER_LENGTH, HOP_LENGTH, FILTER_LENGTH,
        #                          torch.hann_window(FILTER_LENGTH), center=False)
        #
        # mix_spec = torch.stft(batch_info.mix.transpose(1, 0), FILTER_LENGTH, HOP_LENGTH, FILTER_LENGTH,
        #                       torch.hann_window(FILTER_LENGTH), center=False)

        # f = np.arange(0, len(speech_spec[0, :, 0, 0]))
        # t = np.arange(0, len(speech_spec[0, 0, :, 0]))
        # plt.pcolormesh(t, f, np.sqrt(speech_spec[0, :, :, 0] ** 2 + speech_spec[0, :, :, 1] ** 2), cmap='RdGy')
        # plt.title('STFT Magnitude')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.tight_layout()
        # plt.savefig(RESULT_STORE + 'test_clean.jpg')
        # plt.close()

        # f = np.arange(0, len(mix_spec[0, :, 0, 0]))
        # t = np.arange(0, len(mix_spec[0, 0, :, 0]))
        # plt.pcolormesh(t, f, np.sqrt(mix_spec[0, :, :, 0] ** 2 + mix_spec[0, :, :, 1] ** 2), cmap='RdGy')
        # plt.title('STFT Magnitude')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.tight_layout()
        # plt.savefig(RESULT_STORE + 'test_mix.jpg')
        # plt.close()


        # return mix_spec.permute(0, 2, 1, 3), speech_spec.permute(0, 2, 1, 3), batch_info.frame
        return mix_spec,speech_spec,speech_mag,batch_info.frame
