import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram
from torch.autograd import Variable
import scipy.io as scio
import librosa
import numpy as np
from net.net import Net
from load_data.SpeechDataLoad import SpeechDataset, FeatureCreator

from torch.utils.data import DataLoader
import torch.optim as optim
from utils.model_handle import save_model, resume_model
from utils.loss_set import LossHelper,SISDRLoss
from utils.pesq import pesq
import progressbar
from config import *
from utils.util import get_alpha, expandWindow
import time
from utils.stft_istft import STFT
from tensorboardX import SummaryWriter
from utils.label_set import LabelHelper

import pickle


def load_obj(root_dir, name):
    with open(root_dir + name, 'rb') as f:
        return pickle.load(f)


class BatchInfo(object):

    def __init__(self, speech, mix, frame):
        self.speech = speech
        self.mix = mix
        self.frame = frame


def collate(batch):
    """
    将每个batch中的数据pad成一样长，采取补零操作
    切记为@staticmethod方法
    :param batch:input和label的list
    :return:input、label和真实帧长的list
    """
    speech_lst = []
    mix_lst = []
    frame_size_lst = []
    for item in batch:
        speech = item[0]
        mix = item[1]
        speech = speech / torch.max(torch.abs(speech))  # 幅值归一化
        mix = mix / torch.max(torch.abs(mix))
        speech_lst.append(speech)
        mix_lst.append(mix)
        # 计算帧长(（语音长度-窗长度）/帧移)
        frame = (item[0].shape[0] - FILTER_LENGTH) // HOP_LENGTH + 1
        # 存储每句话的真实帧长，用于计算loss
        frame_size_lst.append(frame)
    speech_lst = nn.utils.rnn.pad_sequence(speech_lst)
    mix_lst = nn.utils.rnn.pad_sequence(mix_lst)
    return BatchInfo(speech_lst, mix_lst, frame_size_lst)


def validation(net, path, type):
    net.eval()
    file = os.listdir(path)
    # label_helper = LabelHelper()
    # mse = nn.MSELoss()
    SI_SDR = SISDRLoss()
    sum_loss = 0

    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])
    if type == 0:
        n = 50
    else:
        n = VALIDATION_DATA_NUM
    bar = progressbar.ProgressBar(0, n)
    for i in range(n):
        bar.update(i)
        with torch.no_grad():
            data = load_obj(VALIDATION_DATA_PATH, file[i])
            # 迭代输出需要的文件
            speech = np.array(data['speech'])
            speech = torch.Tensor(speech[np.newaxis, :])
            mix = np.array(data['mix'])
            mix = torch.Tensor(mix[np.newaxis, :])
            speech = speech / torch.max(torch.abs(speech))  # 幅值归一化
            mix = mix / torch.max(torch.abs(mix))
            speech_spec = stft.transform(speech.cuda(CUDA_ID[0]))
            speech_mag = stft.spec_transform(speech_spec)
            mix_spec = stft.transform(mix.cuda(CUDA_ID[0]))

            output = net(mix_spec)
            output_mag = torch.sqrt(output[:, :, :, 0] ** 2 + output[:, :, :, 1] ** 2)
            loss = LossHelper.single_spec_mag_loss(torch.cat([output,output_mag.unsqueeze(-1)],dim=-1), torch.cat([speech_spec,speech_mag.unsqueeze(-1)],dim=-1))
            # loss = mse(output, speech_spec)
            # loss = SI_SDR(output,speech_spec)
            sum_loss += loss.item()
    bar.finish()
    return sum_loss / n


def train(net, epoch, data_loader, optimizer):
    global global_step
    global global_time
    global LR
    writer = SummaryWriter(LOG_STORE)
    feature_creator = FeatureCreator()
    SI_SDR=SISDRLoss()
    # stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])
    bar = progressbar.ProgressBar(0, train_data_set.__len__() // TRAIN_BATCH_SIZE)
    for i in range(epoch):
        if i%2==0 and i!=0:
            if LR > 1e-8:
                LR = LR/2
            optimizer.param_groups[0]['lr'] = LR
        sum_loss = 0
        bar.start()
        for batch_idx, batch_info in enumerate(data_loader):
            mix_spec, speech_spec,speech_mag,frame = feature_creator(batch_info)

            bar.update(batch_idx)
            output = net(mix_spec.cuda(CUDA_ID[0]))
            output_mag = torch.sqrt(output[:, :, :, 0] ** 2 + output[:, :, :, 1] ** 2)
            loss = LossHelper.spec_mag_loss(torch.cat([output,output_mag.unsqueeze(-1)],dim=-1), torch.cat([speech_spec,speech_mag.unsqueeze(-1)],dim=-1), frame)
            # loss = SI_SDR(output,speech_spec)
            # loss = LossHelper.mse_loss(output.cuda(CUDA_ID[0]), speech_spec.cuda(CUDA_ID[0]), frame)
            sum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 10次打一次loss
            if global_step % 10 == 0 and global_step != 0:
                writer.add_scalar('Train/Loss', sum_loss / global_time, global_step)
                # cv_loss = validation(net, VALIDATION_DATA_PATH, 0)
                # writer.add_scalar('Train/CV_Loss', cv_loss, global_step)
                # net.train()
                global_time = 0
                sum_loss = 0
            global_time += 1
            global_step += 1
            # if global_step % 100 == 0 and global_step != 0:
            #     save_model(net, optimizer, loss, models_path=MODEL_STORE + 'middle_store/model_' + str(i) + '_' + str(global_step // 1000) +'.pkl')
        cv_loss = validation(net, VALIDATION_DATA_PATH, 1)
        writer.add_scalar('Train/CV_Loss', cv_loss, global_step)
        net.train()
        save_model(net, optimizer, cv_loss, models_path=MODEL_STORE + 'model_' + str(i) + '.pkl')
        bar.finish()


global_step = 0
global_time = 0

if __name__ == "__main__":
    # 初始化训练集
    train_data_set = SpeechDataset(TRAIN_DATA_PATH)
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=TRAIN_BATCH_SIZE,
                                   shuffle=True,
                                   collate_fn=collate,
                                   num_workers=8,
                                   pin_memory=True
                                   )
    NET = Net()
    NET = NET.cuda(CUDA_ID[0])
    NET = nn.DataParallel(NET)
    # optimizer,loss=resume_model(NET, MODEL_STORE + '/model_2.pkl')
    train_optimizer = optim.Adam(NET.parameters(), lr=LR)
    # train_optimizer.load_state_dict(optimizer)
    # train_optimizer.param_groups[0]['lr'] = LR
    train(NET, EPOCH, train_data_loader, train_optimizer)
