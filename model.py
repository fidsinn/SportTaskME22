# Code dedicated to the Sport Task MediaEval22 
__author__ = "Pierre-Etienne Martin"
__copyright__ = "Copyright (C) 2022 Pierre-Etienne Martin"
__license__ = "CC BY 4.0"
__version__ = "1.0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.batchnorm import _BatchNorm


def flatten_features(x):
    '''
    Flatten Features (all dimensions except the batch dimension)
    '''
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

'''
Batch Normalization 1D for ND tensors
Updated with update of Pytorch v1.11.0
'''
class MyBatchNorm(_BatchNorm): ## Replace nn.BatchNorm3d
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        self.saved_shape = input.shape
        if input.dim() != 2 and input.dim() != 3:
            return input.reshape((input.shape[0], input.shape[1], input[0,0].numel()))

    def forward(self, input):
        input = self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode when the batchsize is greater thhan 1, and in eval mode when buffers are None.
        """
        if self.training and input.size()[0]>1:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        ).reshape(self.saved_shape)

        # if self.training and self.track_running_stats and input.size()[0]>1 and self.num_batches_tracked is not None:
        #     self.num_batches_tracked = self.num_batches_tracked + 1
        #     if self.momentum is None:  # use cumulative moving average
        #         exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        #     else:  # use exponential moving average
        #         exponential_average_factor = self.momentum

        # output = F.batch_norm(
        #     input,
        #     self.running_mean,
        #     self.running_var,
        #     self.weight,
        #     self.bias,
        #     self.training or not self.track_running_stats and input.size()[0]>1,
        #     exponential_average_factor,
        #     self.eps)
        # output = output.reshape(self.saved_shape)

        # return output
'''
Fusion Block
'''

class EarlyFusionBlock(nn.Module):
    def __init__(self, fuse = True, fusion_perc = 0.5, cuda=True):
        super(EarlyFusionBlock, self).__init__()
        
        self.fuse = fuse
        self.fusion_perc = fusion_perc

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, feat_1, feat_2):
        if self.fuse:
            feat_1 = feat_1 * (1 - self.fusion_perc) + feat_2 * self.fusion_perc
        return feat_1
'''
3D Attention Blocks  
'''
class BlockConvReluPool3D(nn.Module):
    def __init__(self, in_dim, out_dim, conv_size=(3,3,3), conv_stride=(1,1,1), conv_padding=(1,1,1), pool_size=(2,2,2), pool_stride=(2,2,2), cuda=True):
        super(BlockConvReluPool3D, self).__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, conv_size, stride=conv_stride, padding=conv_padding)
        self.pool = nn.MaxPool3d(pool_size, stride=pool_stride)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, input):
        return self.pool(F.relu(self.conv(input)))

class ResidualBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, cuda=True):
        super(ResidualBlock3D, self).__init__()

        dim_conv = math.ceil(out_dim/4)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.bn1 = MyBatchNorm(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_dim, dim_conv, 1, 1, bias = False)
        self.bn2 = MyBatchNorm(dim_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim_conv, dim_conv, 3, stride, padding = 1, bias = False)
        self.bn3 = MyBatchNorm(dim_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(dim_conv, out_dim, 1, 1, bias = False)
        self.conv4 = nn.Conv3d(in_dim, out_dim , 1, stride, bias = False)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, input):
        residual = input
        out = self.bn1(input)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.in_dim != self.out_dim) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out


class AttentionModule3D(nn.Module):
    def __init__(self, in_dim, out_dim, size1, size2, size3, cuda=True):
        super(AttentionModule3D, self).__init__()

        self.size1 = tuple(size1.astype(int))
        self.size2 = tuple(size2.astype(int))
        self.size3 = tuple(size3.astype(int))

        self.first_residual_blocks = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.trunk_branches = nn.Sequential(
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda),
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda)
        )

        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.skip1 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block2 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.skip2 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block3 = nn.Sequential(
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda),
        	ResidualBlock3D(in_dim, out_dim, cuda=cuda)
        )

        self.block4 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.block5 = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        self.block6 = nn.Sequential(
        	MyBatchNorm(out_dim),
        	nn.ReLU(inplace=True),
        	nn.Conv3d(out_dim, out_dim , kernel_size = 1, stride = 1, bias = False),
        	MyBatchNorm(out_dim),
        	nn.ReLU(inplace=True),
        	nn.Conv3d(out_dim, out_dim , kernel_size = 1, stride = 1, bias = False),
        	nn.Sigmoid()
        )

        self.final = ResidualBlock3D(in_dim, out_dim, cuda=cuda)

        ## Use GPU
        if cuda:
            self.cuda()


    def forward(self, input):
        input = self.first_residual_blocks(input)
        out_trunk = self.trunk_branches(input)

        # 1st level
        out_pool1 =  self.pool1(input)
        out_block1 = self.block1(out_pool1)
        out_skip1 = self.skip1(out_block1)

        #2sd level
        out_pool2 = self.pool2(out_block1)
        out_block2 = self.block2(out_pool2)
        out_skip2 = self.skip2(out_block2)

        # 3rd level
        out_pool3 = self.pool3(out_block2)
        out_block3 = self.block3(out_pool3)
        out_interp3 = F.interpolate(out_block3, size=self.size3, mode='trilinear', align_corners=True)
        out = out_interp3 + out_skip2

        #4th level
        out_softmax4 = self.block4(out)
        out_interp2 = F.interpolate(out_softmax4, size=self.size2, mode='trilinear', align_corners=True)
        out = out_interp2 + out_skip1

        #5th level
        out_block5 = self.block5(out)
        out_interp1 = F.interpolate(out_block5, size=self.size1, mode='trilinear', align_corners=True)

        #6th level
        out_block6 = self.block6(out_interp1)
        out = (1 + out_block6) * out_trunk

        # Final with Attention added
        out_last = self.final(out)

        return out_last

'''
3D Attention Model  
'''
class CNNAttentionNetV1(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[8,16,32,64,128,256], cuda=True):
        super(CNNAttentionNetV1, self).__init__()

        layers = []
        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx == 0:
                pool_size = [2,2,1]
                pool_stride = [2,2,1]
            else:
                pool_size = [2,2,2]
                pool_stride = [2,2,2]
            layers.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            size_data //= pool_stride
            in_dim = out_dim
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3) 
            if idx>=len(filters)-2:
                layers.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
        self.sequential = nn.Sequential(*layers)
        # (W,H,T)=(5,2,3) - lenght features = 7680
        self.linear1 = nn.Linear(size_data[0]*size_data[1]*size_data[2]*in_dim, size_data[0]*size_data[1]*size_data[2]*in_dim//4)
        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(size_data[0]*size_data[1]*size_data[2]*in_dim//4, n_classes)
        self.final = nn.Softmax(1)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features):
        features = self.sequential(features)
        features = features.view(-1, flatten_features(features))
        features = self.activation(self.linear1(features))
        features = self.linear2(features)
        return self.final(features)

class CNNAttentionNetV2(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2, self).__init__()
        # Per default parameters
        # pool_size = (2,2,2)
        # pool_stride = (2,2,2)
        # conv_size=(3,3,3)
        # conv_stride=(1,1,1)
        # conv_padding=(1,1,1)
        # pool_size=(2,2,2)
        layers = []
        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx < 2: 
                conv_size=(7,5,3)
                conv_padding=(3,2,1)
                pool_size=(4,3,2)
                pool_stride = [4,3,2]
            else: # To finally have similar dimension:20x20x24
                pool_size = [2,2,2]
                pool_stride = [2,2,2]
            layers.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            size_data //= pool_stride
            in_dim = out_dim
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            layers.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
        self.sequential = nn.Sequential(*layers)
        # (W,H,T)=(3,2,2) - lenght features = 6144
        size_linear_src = size_data[0]*size_data[1]*size_data[2]*in_dim
        size_linear_dest = size_linear_src//6
        self.linear1 = nn.Linear(size_linear_src, size_linear_dest)
        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(size_linear_dest, n_classes)
        self.final = nn.Softmax(1)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features):
        features = self.sequential(features)
        features = features.view(-1, flatten_features(features))
        features = self.activation(self.linear1(features))
        features = self.linear2(features)
        return self.final(features)

'''
Attention Net that represents one stream
'''
class CNNAttentionNetV1_Stream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[8,16,32,64,128,256], cuda=True):
        super(CNNAttentionNetV1_Stream, self).__init__()

        layers = []
        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx == 0:
                pool_size = [2,2,1]
                pool_stride = [2,2,1]
            else:
                pool_size = [2,2,2]
                pool_stride = [2,2,2]
            layers.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            size_data //= pool_stride
            in_dim = out_dim
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3) 
            if idx>=len(filters)-2:
                layers.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
        self.sequential = nn.Sequential(*layers)
        # (W,H,T)=(5,2,3) - lenght features = 7680
        self.linear1 = nn.Linear(size_data[0]*size_data[1]*size_data[2]*in_dim, size_data[0]*size_data[1]*size_data[2]*in_dim//4)
        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(size_data[0]*size_data[1]*size_data[2]*in_dim//4, n_classes)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features):
        features = self.sequential(features)
        features = features.view(-1, flatten_features(features))
        features = self.activation(self.linear1(features))
        features = self.linear2(features)
        return features

class CNNAttentionNetV2_Stream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2_Stream, self).__init__()
        
        layers = []
        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx < 2: 
                conv_size=(7,5,3)
                conv_padding=(3,2,1)
                pool_size=(4,3,2)
                pool_stride = [4,3,2]
            else: # To finally have similar dimension:20x20x24
                pool_size = [2,2,2]
                pool_stride = [2,2,2]
            layers.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            size_data //= pool_stride
            in_dim = out_dim
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            layers.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
        self.sequential = nn.Sequential(*layers)
        # (W,H,T)=(3,2,2) - lenght features = 6144
        size_linear_src = size_data[0]*size_data[1]*size_data[2]*in_dim
        size_linear_dest = size_linear_src//6
        self.linear1 = nn.Linear(size_linear_src, size_linear_dest)
        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(size_linear_dest, n_classes)

        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features):
        features = self.sequential(features)
        features = features.view(-1, flatten_features(features))
        features = self.activation(self.linear1(features))
        features = self.linear2(features)
        return features

class CNNAttentionNetV1_TwoStream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[8,16,32,64,128,256], cuda=True):
        super(CNNAttentionNetV1_TwoStream, self).__init__()
        
        # need to vopy value of size data as model chabges  it 
        self.stream_one = CNNAttentionNetV1_Stream(size_data.copy(), n_classes)
        self.stream_two = CNNAttentionNetV1_Stream(size_data.copy(), n_classes)

        self.final = nn.Softmax(1)
        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features_s1, features_s2):
        s1_out = self.stream_one(features_s1)
        s2_out = self.stream_two(features_s2)
        return self.final(s1_out + s2_out)

class CNNAttentionNetV2_TwoStream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2_TwoStream, self).__init__()
        
        
        # need to vopy value of size data as model chabges  it 
        self.stream_one = CNNAttentionNetV2_Stream(size_data.copy(), n_classes)
        self.stream_two = CNNAttentionNetV2_Stream(size_data.copy(), n_classes)

        self.final = nn.Softmax(1)
        ## Use GPU
        if cuda:
            self.cuda()

    def forward(self, features_s1, features_s2):
        s1_out = self.stream_one(features_s1)
        s2_out = self.stream_two(features_s2)
        return self.final(s1_out + s2_out)


class CNNAttentionNetV2L_TwoStream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2L_TwoStream, self).__init__()
        layers_s1 = []
        layers_s2 = []

        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx < 2: 
                pool_size=(4,3,2)
                pool_stride = [4,3,2]
            else: # To finally have similar dimension:20x20x24
                pool_size = [2,2,2]
                pool_stride = [2,2,2]

            layers_s1.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            layers_s2.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))

            size_data //= pool_stride
            in_dim = out_dim
            
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            layers_s1.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
            layers_s2.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))

        self.sequential_s1 = nn.Sequential(*layers_s1)
        self.sequential_s2 = nn.Sequential(*layers_s2)

        # (W,H,T)=(3,2,2) - lenght features = 6144
        size_linear_src = size_data[0]*size_data[1]*size_data[2]*in_dim
        size_linear_dest = size_linear_src//6
        self.linear1_s1 = nn.Linear(size_linear_src, size_linear_dest)
        self.linear1_s2 = nn.Linear(size_linear_src, size_linear_dest)
        self.activation_s1 = nn.ReLU()
        self.activation_s2 = nn.ReLU()

        self.linear2_s1 = nn.Linear(size_linear_dest, n_classes)
        self.linear2_s2 = nn.Linear(size_linear_dest, n_classes)

        self.linear_fuse = nn.Linear(n_classes * 2, n_classes)

        self.final = nn.Softmax(1)

        ## Use GPU
        if cuda:
            self.cuda()


    def forward(self, features_s1, features_s2):

        features_s1 = self.sequential_s1(features_s1)
        features_s2 = self.sequential_s2(features_s2)

        features_s1 = features_s1.view(-1, flatten_features(features_s1))
        features_s2 = features_s2.view(-1, flatten_features(features_s2))

        features_s1 = self.activation_s1(self.linear1_s1(features_s1))
        features_s2 = self.activation_s2(self.linear1_s2(features_s2))

        features_s1 = self.linear2_s1(features_s1)
        features_s2 = self.linear2_s2(features_s2)

        feature_final = torch.cat((features_s1,features_s2), dim=1)
        feature_final = self.linear_fuse(feature_final)

        return self.final(feature_final)

class CNNAttentionNetV2M_TwoStream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2M_TwoStream, self).__init__()
        layers_s1 = []
        layers_s2 = []

        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx < 2: 
                pool_size=(4,3,2)
                pool_stride = [4,3,2]
            else: # To finally have similar dimension:20x20x24
                pool_size = [2,2,2]
                pool_stride = [2,2,2]

            layers_s1.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            layers_s2.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))

            size_data //= pool_stride
            in_dim = out_dim
            
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            layers_s1.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
            layers_s2.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))

        self.sequential_s1 = nn.Sequential(*layers_s1)
        self.sequential_s2 = nn.Sequential(*layers_s2)

        # (W,H,T)=(3,2,2) - lenght features = 6144
        size_linear_src = size_data[0]*size_data[1]*size_data[2]*in_dim
        size_linear_dest = size_linear_src//6
        self.linear1 = nn.Linear(size_linear_src*2, size_linear_dest)
        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(size_linear_dest, n_classes)

        self.final = nn.Softmax(1)

        ## Use GPU
        if cuda:
            self.cuda()


    def forward(self, features_s1, features_s2):

        features_s1 = self.sequential_s1(features_s1)
        features_s2 = self.sequential_s2(features_s2)

        features_s1 = features_s1.view(-1, flatten_features(features_s1))
        features_s2 = features_s2.view(-1, flatten_features(features_s2))

        features = torch.cat((features_s1,features_s2), dim=1)
        
        features = self.activation(self.linear1(features))

        features = self.linear2(features)


        return self.final(features)
        
# continuas fusion after every layer

class CNNAttentionNetV2C_TwoStream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2C_TwoStream, self).__init__()
        self.layers_s1 = []
        self.layers_s2 = []

        for idx, out_dim in enumerate(filters):
            # First layer, no diminution of dimension on temporal domain, double
            if idx < 2: 
                conv_size=(7,5,3)
                conv_padding=(3,2,1)
                pool_size=(4,3,2)
                pool_stride = [4,3,2]
            else: # To finally have similar dimension:20x20x24
                pool_size = [2,2,2]
                pool_stride = [2,2,2]

            self.layers_s1.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            self.layers_s2.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))

            size_data //= pool_stride
            in_dim = out_dim
            
            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            self.layers_s1.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
            self.layers_s2.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))

        # self.sequential_s1 = nn.Sequential(*layers_s1)
        # self.sequential_s2 = nn.Sequential(*layers_s2)

        # (W,H,T)=(3,2,2) - lenght features = 6144
        size_linear_src = size_data[0]*size_data[1]*size_data[2]*in_dim
        size_linear_dest = size_linear_src//6

        self.linear1_s1 = nn.Linear(size_linear_src, size_linear_dest)
        self.linear1_s2 = nn.Linear(size_linear_src, size_linear_dest)

        self.activation_s1 = nn.ReLU()
        self.activation_s2 = nn.ReLU()

        self.linear2_s1 = nn.Linear(size_linear_dest, n_classes)
        self.linear2_s2 = nn.Linear(size_linear_dest, n_classes)

        self.final = nn.Softmax(1)

        ## Use GPU
        if cuda:
            self.cuda()

    def fusion(self, features_s1, features_s2, fusion_perc = 0.3):
        fused = features_s1 * (1 - fusion_perc) + features_s2 * fusion_perc
        return fused

    def forward(self, features_s1, features_s2):

        for i, layer in enumerate(self.layers_s1):            
            if i == 0:
                features_s1 = layer(features_s1)
            else:
                features_s1 = layer(features_s1 + features_s2)
            
            features_s2 = self.layers_s2[i](features_s2)


        features_s1 = features_s1.view(-1, flatten_features(features_s1 + features_s2))
        features_s2 = features_s2.view(-1, flatten_features(features_s2))

        features_s1 = self.activation_s1(self.linear1_s1(features_s1 + features_s2))
        features_s2 = self.activation_s2(self.linear1_s2(features_s2))

        features_s1 = self.linear2_s1(features_s1 + features_s2)
        features_s2 = self.linear2_s2(features_s2)


        return self.final(features_s1 + features_s2)



class CNNAttentionNetV2E_TwoStream(nn.Module):
    def __init__(self, size_data, n_classes, in_dim=3, filters=[32,64,128,256,512], cuda=True):
        super(CNNAttentionNetV2E_TwoStream, self).__init__()
        layers_stream1_bf = []
        layers_stream2_bf = []
        layers_stream1_af = []
        layers_stream2_af = []

        # fusion index
        f_idx = 2

        # layers before fusion
        for idx, out_dim in enumerate(filters[:f_idx]):

            # First layer, no diminution of dimension on temporal domain, double
            if idx < 2: 
                pool_size=(4,3,2)
                pool_size = [2,2,2]
                pool_stride = [2,2,2]

            layers_stream1_bf.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            layers_stream2_bf.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))

            size_data //= pool_stride
            in_dim = out_dim

            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            layers_stream1_bf.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
            layers_stream2_bf.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))

        # layers after fusion
        for idx, out_dim in enumerate(filters[f_idx:]):
            # First layer, no diminution of dimension on temporal domain, double
            if idx < 1: 
                pool_size=(4,3,2)
                pool_stride = [4,3,2]
            else: # To finally have similar dimension:20x20x24
                pool_size = [2,2,2]
                pool_stride = [2,2,2]

            layers_stream1_af.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))
            layers_stream2_af.append(BlockConvReluPool3D(in_dim, out_dim, cuda=cuda, pool_size=pool_size, pool_stride=pool_stride))

            size_data //= pool_stride
            in_dim = out_dim

            # No attention mechanism on the two last layers (min dim = 2 in this configuration) - (W,H,T)=(5,2,3)
            layers_stream1_af.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))
            layers_stream2_af.append(AttentionModule3D(in_dim, in_dim, size_data, np.ceil(size_data/2), np.ceil(size_data/4), cuda=cuda))

        self.sequential_stream1_bf = nn.Sequential(*layers_stream1_bf)
        self.sequential_stream2_bf = nn.Sequential(*layers_stream2_bf)

        self.sequential_stream1_af = nn.Sequential(*layers_stream1_af)
        self.sequential_stream2_af = nn.Sequential(*layers_stream2_af)

        # (W,H,T)=(3,2,2) - lenght features = 6144
        size_linear_src = size_data[0]*size_data[1]*size_data[2]*in_dim
        size_linear_dest = size_linear_src//6
        if cuda:
            self.cuda()

    # unidirectional fusion s2 in s1
    def fusion(self, features_s1, features_s2, fusion_perc = 0.7):
        s1 = features_s1 * (1 - fusion_perc) + features_s2 * fusion_perc
        s2 = features_s2
        return s1, s2


    def forward(self, features_s1, features_s2):
        features_s1 = self.sequential_stream1_bf(features_s1)
        features_s2 = self.sequential_stream2_bf(features_s2)

        # does this affect the attention blocks?
        features_s1, features_s2 = self.fusion(features_s1, features_s2)

        features_s1 = self.sequential_stream1_af(features_s1)
        features_s2 = self.sequential_stream2_af(features_s2)

        features_s1 = features_s1.view(-1, flatten_features(features_s1))
        features_s2 = features_s2.view(-1, flatten_features(features_s2))

        features_s1 = self.activation_stream1(self.linear1_stream1(features_s1))
        features_s2 = self.activation_stream2(self.linear1_stream2(features_s2))
        features_s1 = self.linear2(features_s1)
        features_s2 = self.linear2(features_s2)

        return self.final(features_s1 + features_s2)