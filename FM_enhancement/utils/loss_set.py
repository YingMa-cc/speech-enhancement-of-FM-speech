import torch
from torch.nn.utils.rnn import pad_sequence
from config import *
from utils.util import apply_reduction


class LossHelper(object):

    @staticmethod
    def mse_loss(est, label, nframes):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :param nframes: 每个batch中的真实帧长
        :return:loss
        """
        with torch.no_grad():
            mask_for_loss_list = []
            # 制作掩码
            for frame_num in nframes:
                mask_for_loss_list.append(
                    torch.ones(frame_num, label.size()[2], 2, dtype=torch.float32, device=CUDA_ID[0]))
            # input: list of tensor
            # output: B T *
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda(CUDA_ID[0])

        # 使用掩码计算真实值
        masked_est = est * mask_for_loss
        masked_label = label * mask_for_loss
        loss = ((masked_est - masked_label) ** 2).sum() / mask_for_loss.sum()
        return loss

    @staticmethod
    def spec_mag_loss(est, label, nframes):
        """
        计算spectrogram和magnitude联合l1loss
        :param est: 网络输出
        :param label: label
        :param nframes: 真实帧长
        :return: loss
        """
        with torch.no_grad():
            mask_for_loss_list = []
            # 制作掩码
            for frame_num in nframes:
                mask_for_loss_list.append(
                    torch.ones(frame_num, label.size()[2], 3, dtype=torch.float32, device=CUDA_ID[0]))
            # input: list of tensor
            # output: B T *
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda(CUDA_ID[0])

        # 使用掩码计算真实值
        masked_est = est * mask_for_loss
        masked_label = label * mask_for_loss
        loss = (abs(masked_est - masked_label)).sum()/ (mask_for_loss.sum()+EPSILON)
        return loss


    @staticmethod
    def single_spec_mag_loss(est, label):

        loss = (abs(est - label)).sum()/(torch.ones(est.size()).sum()+EPSILON)
        return loss

class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.
    Note that this returns the negative of the SI-SDR loss.
    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)
    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=False, eps=1e-8, reduction='mean'):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum() / ((target ** 2).sum())
        target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10((target**2).sum()/((res**2).sum() + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses


