from audioop import reverse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
from compressai.layers import *
from compressai.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention

class AttModule(nn.Module):
    def __init__(self, N):
        super(AttModule, self).__init__()
        self.forw_att = AttentionBlock(N)
        self.back_att = AttentionBlock(N)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_att(x)
        else:
            return self.back_att(x)

class WinAttModule(nn.Module):
    def __init__(self, N):
        super(WinAttModule, self).__init__()
        self.forw_att = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)
        self.back_att = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_att(x)
        else:
            return self.back_att(x)

class EnhModule(nn.Module):
    def __init__(self, nf):
        super(EnhModule, self).__init__()
        self.forw_enh = EnhBlock(nf)
        self.back_enh = EnhBlock(nf)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_enh(x)
        else:
            return self.back_enh(x)

class EnhBlock(nn.Module):
    def __init__(self, nf):
        super(EnhBlock, self).__init__()
        self.layers = nn.Sequential(
            DenseBlock(3, nf),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            DenseBlock(nf, 3)
        )

    def forward(self, x):
        return x + self.layers(x) * 0.2


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CFEMEnhModule(nn.Module):
    def __init__(self, nf):
        super(CFEMEnhModule, self).__init__()
        self.forw_enh = CFEMEnhBlock(nf, nf,stride=1, scale=0.1, groups=8, dilation=1, thinning=4)
        self.back_enh = CFEMEnhBlock(nf,nf,stride=1, scale=0.1, groups=8, dilation=1, thinning=4)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_enh(x)
        else:
            return self.back_enh(x)

class CFEMEnhBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1, scale = 0.1, groups=8, thinning=2, k = 7, dilation=1):
        super(CFEMEnhBlock, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        second_in_planes = in_planes // thinning

        p = (k-1)//2
        self.cfem_a = list()
        self.cfem_a += [DenseBlock(3, in_planes)]
        self.cfem_a += [BasicConv(in_planes, in_planes, kernel_size = (1,k), stride = 1, padding = (0,p), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size=3, stride=stride, padding=dilation, groups = 4, dilation=dilation)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = (k, 1), stride = 1, padding = (p, 0), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        # self.cfem_a += [DenseBlock(second_in_planes, 3)]
        self.cfem_a = nn.ModuleList(self.cfem_a)

        self.cfem_b = list()
        self.cfem_b += [DenseBlock(3, in_planes)]
        self.cfem_b += [BasicConv(in_planes, in_planes, kernel_size = (k,1), stride = 1, padding = (p,0), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 3, stride=stride, padding=dilation,groups =4,dilation=dilation)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = (1, k), stride = 1, padding = (0, p), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        # self.cfem_b += [DenseBlock(second_in_planes, 3)]

        self.cfem_b = nn.ModuleList(self.cfem_b)


        self.ConvLinear = BasicConv(2 * second_in_planes, 3, kernel_size = 1, stride = 1, relu = False)
        self.shortcut = BasicConv(3, 3, kernel_size = 1, stride = stride, relu = False)
        self.relu = nn.ReLU(inplace = False)

    def forward(self,x):
        x1 = self.cfem_a[0](x)
        x1 = self.cfem_a[1](x1)
        x1 = self.cfem_a[2](x1)
        x1 = self.cfem_a[3](x1)
        x1 = self.cfem_a[4](x1)
        x1 = self.cfem_a[5](x1)


        x2 = self.cfem_b[0](x)
        x2 = self.cfem_b[1](x2)
        x2 = self.cfem_b[2](x2)
        x2 = self.cfem_b[3](x2)
        x2 = self.cfem_b[4](x2)
        x2 = self.cfem_b[5](x2)


        out = torch.cat([x1, x2], 1)
        out = self.ConvLinear(out)
        #//TODO - try out with adding short cut. Uncoment below line``````````````````````````````````````````````````````````````````````````````````````````````````````
        out = out * self.scale

        out = self.relu(out)
        out=out+x
        return out

    def get_CFEM(cfe_type='large', in_planes=512, out_planes=512, stride=1, scale=1, groups=8, dilation=1):
        assert cfe_type in ['large', 'normal', 'light'], 'no that type of CFEM'
        if cfe_type == 'large':
            return CFEMEnhBlock(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=2)
        elif cfe_type == 'normal':
            return CFEMEnhBlock(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=4)
        else:
            return CFEMEnhBlock(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=8)



class RFBModule(nn.Module):
    def __init__(self, nf):
        super(RFBModule, self).__init__()
        # print(f'nf: {nf}')
        self.forw_rfb = ReceptiveFieldBlock(nf, nf)
        self.back_rfb = ReceptiveFieldBlock(nf, nf)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_rfb(x)
        else:
            return self.back_rfb(x)

class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, clamp=1.0):
        """Modules introduced in RFBNet paper.
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """

        super(ReceptiveFieldBlock, self).__init__()
        self.split_len1 = in_channels // 4
        self.out_channels = 3 * in_channels // 4 
        branch_channels = in_channels // 4
        self.clamp = clamp
        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (1, 1), dilation=1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, (branch_channels) * 3, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d((branch_channels) * 3, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (5, 5), dilation=5),
        )
        self.conv_linear = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: torch.Tensor, rev=False) -> torch.Tensor:
        print(self.split_len1, self.out_channels)
        if(rev == False):
            print(f'branch1 i/p: {x.shape}')
            branch1 = self.branch1(x)
            print(f'branch2 i/p: {x.shape}')
            branch2 = self.branch2(x)
            print(f'branch3 i/p: {x.shape}')
            branch3 = self.branch3(x)
            print(f'branch4 i/p: {x.shape}')
            branch4 = self.branch4(x)
        else:
            branch4 = self.branch4(x)
            branch3 = self.branch3(x)
            branch2 = self.branch2(x)
            branch1 = self.branch1(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        
        out = x + out
        return out

class ReceptiveFieldDenseModule(nn.Module):
    def __init__(self, nf):
        super(ReceptiveFieldDenseModule, self).__init__()
        # print(f'nf: {nf}')
        self.forw_rfb = ReceptiveFieldDenseBlock(nf, nf)
        self.back_rfb = ReceptiveFieldDenseBlock(nf, nf)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_rfb(x)
        else:
            return self.back_rfb(x)

class ReceptiveFieldDenseBlock(nn.Module):
    """Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
    RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, channels: int, growth_channels: int):
        """
        Args:
            channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
        """

        super(ReceptiveFieldDenseBlock, self).__init__()
        self.rfb1 = ReceptiveFieldBlock(channels + 0 * growth_channels, growth_channels)
        self.rfb2 = ReceptiveFieldBlock(channels + 1 * growth_channels, growth_channels)
        self.rfb3 = ReceptiveFieldBlock(channels + 2 * growth_channels, growth_channels)
        self.rfb4 = ReceptiveFieldBlock(channels + 3 * growth_channels, growth_channels)
        self.rfb5 = ReceptiveFieldBlock(channels + 4 * growth_channels, channels)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor, rev=False) -> torch.Tensor:
        identity = x
        if (rev==False):
            rfb1 = self.leaky_relu(self.rfb1(x))
            rfb2 = self.leaky_relu(self.rfb2(torch.cat([x, rfb1], 1)))
            rfb3 = self.leaky_relu(self.rfb3(torch.cat([x, rfb1, rfb2], 1)))
            rfb4 = self.leaky_relu(self.rfb4(torch.cat([x, rfb1, rfb2, rfb3], 1)))
            rfb5 = self.identity(self.rfb5(torch.cat([x, rfb1, rfb2, rfb3, rfb4], 1)))
            out = torch.mul(rfb5, 0.2)
            
        else:
            rfb5 = self.leaky_relu(self.rfb5(x))
            rfb4 = self.leaky_relu(self.rfb4(torch.cat([x, rfb5], 1)))
            rfb3 = self.leaky_relu(self.rfb3(torch.cat([x, rfb5, rfb4], 1)))
            rfb2 = self.leaky_relu(self.rfb2(torch.cat([x, rfb5, rfb4, rfb3], 1)))
            rfb1 = self.identity(self.rfb1(torch.cat([x, rfb5, rfb4, rfb3, rfb2], 1)))
            out = torch.div(rfb1, 0.2)
        out = torch.add(out, identity)

        return out

# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    def __init__(self, channels: int, growths: int):
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.rfdb1 = ReceptiveFieldDenseBlock(channels, growths)
        self.rfdb2 = ReceptiveFieldDenseBlock(channels, growths)
        self.rfdb3 = ReceptiveFieldDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rfdb1(x)
        out = self.rfdb2(out)
        out = self.rfdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class cnn(nn.Module):
    def __init__(self, M):
        super(cnn, self).__init__()
        self.in_nc = 3
        self.out_nc = M
        N=M
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x, rev=False):
        if not rev:
            x=self.g_a(x)
            b, c, h, w = x.size()
            x = torch.mean(x.view(b, c//self.out_nc, self.out_nc, h, w), dim=1)
        else:
            
            x=self.g_s(x)
        return x
    

# class InvComp(nn.Module):
#     def __init__(self, M):
#         super(InvComp, self).__init__()
#         self.in_nc = 3
#         self.out_nc = M
#         self.operations1 = nn.ModuleList()
#         self.operations2 = nn.ModuleList()
#         self.operations3 = nn.ModuleList()
#         self.operations4 = nn.ModuleList()



#         # 1st level
#         b1 = SqueezeLayer(2)
#         self.operations1.append(b1)
#         self.in_nc *= 4
#         b1 = InvertibleConv1x1(self.in_nc)
#         self.operations1.append(b1)
#         b1 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
#         self.operations1.append(b1)
#         b1 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
#         self.operations1.append(b1)
#         b1 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
#         self.operations1.append(b1)
#         # 2nd level
#         b2 = SqueezeLayer(2)
#         self.operations2.append(b2)
#         self.in_nc *= 4
#         b2 = InvertibleConv1x1(self.in_nc)
#         self.operations2.append(b2)
#         b2 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
#         self.operations2.append(b2)
#         b2 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
#         self.operations2.append(b2)
#         b2 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
#         self.operations2.append(b2)
#         self.attn1 = WinAttModule(self.in_nc)


#         # 3rd level
#         b3 = SqueezeLayer(2)
#         self.operations3.append(b3)
#         self.in_nc *= 4
#         b3 = InvertibleConv1x1(self.in_nc)
#         self.operations3.append(b3)
#         b3 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
#         self.operations3.append(b3)
#         b3 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
#         self.operations3.append(b3)
#         b3 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
#         self.operations3.append(b3)

#         # 4th level
#         b4 = SqueezeLayer(2)
#         self.operations4.append(b4)
#         self.in_nc *= 4
#         b4 = InvertibleConv1x1(self.in_nc)
#         self.operations4.append(b4)
#         b4 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
#         self.operations4.append(b4)
#         b4 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
#         self.operations4.append(b4)
#         b4 = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
#         self.operations4.append(b4)
#         self.attn2 = WinAttModule(self.in_nc)

#     def forward(self, x, rev=False):
#         if not rev:
#             for op in self.operations1:
#                 x = op.forward(x, False)
#             for op in self.operations2:
#                 x = op.forward(x, False)
#             x = self.attn1(x, rev=False)
#             for op in self.operations3:
#                 x = op.forward(x, False)
#             for op in self.operations4:
#                 x = op.forward(x, False)
#             x = self.attn2(x, rev=False)
#             b, c, h, w = x.size()
#             x = torch.mean(x.view(b, c//self.out_nc, self.out_nc, h, w), dim=1)
#         else:
            
#             times = self.in_nc // self.out_nc
            
#             x = x.repeat(1, times, 1, 1)
#             x = self.attn2(x, rev=True)
#             for op in reversed(self.operations4):
#                 x = op.forward(x, True)
#             for op in reversed(self.operations3):
#                 x = op.forward(x, True)
#             x = self.attn1(x, rev=True)
#             for op in reversed(self.operations2):
#                 x = op.forward(x, True)
#             for op in reversed(self.operations1):
#                 x = op.forward(x, True)
#         return x

class CouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1) )) + self.H2(x2)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1) )) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1) ))
            y1 = (x1 - self.H2(y2)).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1) ))
        return torch.cat((y1, y2), 1)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights(self.conv3, 0)
        
    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        return conv3

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, reverse=False):
        if not reverse:
            output = self.squeeze2d(input, self.factor)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input, self.factor)
            return output
        
    def jacobian(self, x, rev=False):
        return 0
        
    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

# class InvertibleConv1x1(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
#         w_shape = [num_channels, num_channels]
#         w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
#         self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
#         self.w_shape = w_shape

#     def get_weight(self, input, reverse):
#         w_shape = self.w_shape
#         if not reverse:
#             weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
#         else:
#             weight = torch.inverse(self.weight.double()).float() \
#                 .view(w_shape[0], w_shape[1], 1, 1)
#         return weight

#     def forward(self, input, reverse=False):
#         weight = self.get_weight(input, reverse)
#         if not reverse:
#             z = F.conv2d(input, weight)
#             return z
#         else:
#             z = F.conv2d(input, weight)
#             return z

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu", train=False):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block_simplified, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = IRes_Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 1
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, padding=0),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size3 = 1
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=0),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch, train=train)
        else:
            self.actnorm = None

    def forward(self, x, rev=False, ignore_logdet=False, maxIter=25):
        if not rev:
            """ bijective or injective block forward """
            if self.stride == 2:
                x = self.squeeze.forward(x)
            if self.actnorm is not None:
                x, an_logdet = self.actnorm(x)
            else:
                an_logdet = 0.0
            Fx = self.bottleneck_block(x)
            if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
                trace = torch.tensor(0.)
            else:
                trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
            y = Fx + x
            return y, trace + an_logdet
        else:
            y = x
            for iter_index in range(maxIter):
                summand = self.bottleneck_block(x)
                x = y - summand

            if self.actnorm is not None:
                x = self.actnorm.inverse(x)
            if self.stride == 2:
                x = self.squeeze.inverse(x)
            return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)