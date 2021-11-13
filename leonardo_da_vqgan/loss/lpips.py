"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
import os
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from leonardo_da_vqgan.utils.utils import get_ckpt_path

# self normalize the tensor
def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

# take the mean across the pixel dimensions
def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

# single linear layer, 1x1 conv operation
class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, input):
        # update values in input by shift and scale
        return (input - self.shift) / self.scale

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # get pretrained vgg16 features
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        
        # 30 features spread out over 5 slices
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        
        # 4 features
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # 5 features
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # 7 features
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # 7 features
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # 7 features
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # don't update params if required_grad=False
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # pass input through first layer (4 features)
        h = self.slice1(X)
        h_relu1_2 = h # save result

        # pass output of first layer through second
        h = self.slice2(h)
        h_relu2_2 = h # save result

        # pass output of second layer through third
        h = self.slice3(h)
        h_relu3_3 = h

        # pass output of third layer through fourth
        h = self.slice4(h)
        h_relu4_3 = h
        
        # pass output of fourth layer through final
        h = self.slice5(h)
        h_relu5_3 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False
    
    # load the pretrained vgg16 model
    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    #load the pretrained weights again and return a new model
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model
    
    def forward(self, input, target):
        # scale the input and the target
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        # pass the scaled input and target through vgg16 
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        # set up feats comparison
        feats0, feats1, diffs = {}, {}, {}

        # for each linear convolution
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        # for each channel number for each layer (layer1=64,layer2=128,...)
        for kk in range(len(self.chns)):
            # feats from that layer stored as normalized outputs
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            # save difference of feats from input and target as diff squared
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # result = spacial average of each linear layer's activation due to calculated feature difference
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]

        # return sum of spacial averages 
        val = res[0] 
        for l in range(1, len(self.chns)):
            val += res[l]
        return val