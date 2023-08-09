from pvt import pvt_large
from torchvision.models import resnet34
import torch.nn as nn
import torch
import torch.nn.functional as F

class PAM(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( N X C X H X W)
        returns :
            out : attention value + input feature
            attention: N X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # B => N, C, HW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)
        # B' => N, HW, C
        proj_query = proj_query.permute(0, 2, 1)
        # C => N, C, HW
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # B'xC => N, HW, HW
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(B'xC) => N, HW, HW
        attention = self.softmax(energy)

        # D => N, C, HW
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # DxS' => N, C, HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # N, C, H, W
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( N X C X H X W)
        returns :
            out : attention value + input feature
            attention: N X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # N, C, C, bmm 批次矩阵乘法
        energy = torch.bmm(proj_query, proj_key)

        # 这里实现了softmax用最后一维的最大值减去了原始数据, 获得了一个不是太大的值
        # 沿着最后一维的C选择最大值, keepdim保证输出和输入形状一致, 除了指定的dim维度大小为1
        energy_new = torch.max(energy, -1, keepdim=True)
        energy_new = energy_new[0].expand_as(energy)  # 复制的形式扩展到energy的尺寸
        energy_new = energy_new - energy
        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class ATT(nn.Module):  # BAM
    def __init__(self, channels):
        super(ATT, self).__init__()
        self.pam = PAM(channels)  # channel attention branch
        self.cam = CAM()  # spatial attention branch
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        _, c, h, w = x.shape  # b,c,h,w
        channel_weights = self.pam(x)  # (B,C,1,1)
        spatial_weights = self.cam(x)  # (B,1,H,W)
        # (B,C,1,1) + (B,1,H,W) -> (B,C,H,W)
        weights = self.sigmoid(channel_weights + spatial_weights)  # M(F) = σ(Mc(F)+Ms(F))，对应原文公式(2)
        return x + x * weights  # F0 = F+F⊗M(F)，对应原文公式(1)
    
class Dualconv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Dualconv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class u_trans(nn.Module):
    def __init__(self, n_channels, num_classes,pretrained=True, drop=0., 
                 dim=[7, 14, 28, 56, 112],filters=[64, 128, 256, 512, 1024, 2048, 2560, 1280,320]):
                                          #         0,   1,   2,   3,   4    5     6     7    8
        super(u_trans, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.dim = dim
        self.filters = filters
        
        self.dualconv4 = Dualconv(filters[3]+filters[3], filters[3], filters[3])
        self.dualconv3 = Dualconv(filters[3]+filters[3], filters[3], filters[2])
        self.dualconv2 = Dualconv(filters[2]+filters[2], filters[2], filters[1])
        self.dualconv1 = Dualconv(filters[0]+filters[1], filters[0], filters[0])

        backbone = resnet34(pretrained=True)
        self.layer1 = backbone.layer1
        self.emb2 = backbone.layer2
        self.emb3 = backbone.layer3
        self.emb4 = backbone.layer4

        # self.pvt_small = pvt_small()
        # device = torch.device('cuda:0')
        model = pvt_large()
        self.model = model
        
        self.att2 = ATT(filters[1] + filters[1])
        self.att3 = ATT(filters[2] + filters[2])
        self.att4 = ATT(filters[3] + filters[3])
        self.up1 = nn.ConvTranspose2d(filters[3], filters[3], 2, 2)
        self.up2 = nn.ConvTranspose2d(filters[2], filters[2], 2, 2)
        self.up3 = nn.ConvTranspose2d(filters[1], filters[1], 2, 2)
        self.finalup = nn.ConvTranspose2d(filters[0], num_classes, 4, 4)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        pvt_out_list, blocks_list = self.model(x)
        # print(blocks_list[1])
        s1 = pvt_out_list[0]  # s1->64*56*56元组
        print(s1.shape)
        
        l2 = self.emb2(s1).flatten(2).transpose(1, 2)   # 28*28*128
        # l2 = nn.LayerNorm(l2.shape[2])
        # print(l2.shape)# 2*128*28*28
        for module in blocks_list[1]:
            s2 = module(l2, 28, 28)  # s2->2*（28*28）*128
        l2 = l2.view(l2.shape[0], 28, 28, l2.shape[-1]).permute(0, 3, 1, 2) 
        s2 = s2.view(s2.shape[0], 28, 28, s2.shape[-1]).permute(0, 3, 1, 2) 
        cat_dual_2 = self.att2(torch.cat((l2, s2), dim=1))  # 28*28*(128+128)=256

        l3 = self.emb3(s2).flatten(2).transpose(1, 2)  # 14*14*256
        # l3 = nn.LayerNorm(l3.shape[2])
        # print(l3.shape)# 2*256*14*14
        for module in blocks_list[2]:
            s3 = module(l3, 14, 14)  # s2->2*（14*14）*256
        l3 = l3.view(l3.shape[0], 14, 14, l3.shape[-1]).permute(0, 3, 1, 2) 
        s3 = s3.view(s3.shape[0], 14, 14, s3.shape[-1]).permute(0, 3, 1, 2) 
        cat_dual_3 = self.att3(torch.cat((l3, s3), dim=1))  # 14*14*(256+256)=512

        l4 = self.emb4(s3).flatten(2).transpose(1, 2)   # 7*7*512
        # l4 = nn.LayerNorm(l4.shape[2])
        for module in blocks_list[3]:
            s4 = module(l4, 7, 7)  # s2->2*（14*14）*512
        l4 = l4.view(l4.shape[0], 7, 7, l4.shape[-1]).permute(0, 3, 1, 2) 
        s4 = s4.view(s4.shape[0], 7, 7, s4.shape[-1]).permute(0, 3, 1, 2) 
        cat_dual_4 = self.att4(torch.cat((l4, s4), dim=1))  # 7*7*(512+512)=1024
        cat_dual_4 = self.dualconv4(cat_dual_4)  # 7*7*512
        cat_dual_4 = self.up1(cat_dual_4)  # 14*14*512

        cat_three_3 = torch.cat((cat_dual_4, cat_dual_3), dim=1)  # 14*14*(512+512)=1024
        cat_three_3 = self.dualconv3(cat_three_3)  # 14*14*256
        cat_three_3 = self.up2(cat_three_3)  # 28*28*256

        cat_three_2 = torch.cat((cat_three_3, cat_dual_2), dim=1)  # 28*28*(256+128)=1280
        cat_three_2 = self.dualconv2(cat_three_2)  # 28*28*256
        cat_three_2 = self.up3(cat_three_2)  # 56*56*128

        cat_three_1 = torch.cat((cat_three_2, s1),dim=1)# 56*56*(128+64)=512
        cat_three_1 = self.dualconv1(cat_three_1) # 56*56*64

        x = self.finalup(cat_three_1)

        return x

if __name__=="__main__":
    x = torch.randn(2, 1, 224, 224)
    net = u_trans(1, 3)
    print(net(x).shape)




