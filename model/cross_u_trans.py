import torch
from torchvision import models as resnet_model
from torch import nn
from model.transformer import TransformerModel

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class cross_u_trans(nn.Module):
    def __init__(self, n_channels=3, num_classes=9, heads=8, depth=3, dim=320):
        super(cross_u_trans, self).__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.heads = heads
        self.depth = depth
        self.dim = dim

        resnet = resnet_model.resnet34(pretrained=True)
        self.firstconv = resnet.conv1  # 224x224
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.cnn_1 = resnet.layer1  # 64    ,112
        self.cnn_2 = resnet.layer2  # 128   ,56
        self.cnn_3 = resnet.layer3  # 256   ,28
        self.cnn_4 = resnet.layer4  # 512   ,14

        self.vit_1 = TransformerModel(dim=128, mlp_dim=2 * 128 , heads=heads, depth=depth, sr_ratio=4)
        self.vit_2 = TransformerModel(dim=256, mlp_dim=2 * 256, heads=heads, depth=depth, sr_ratio=2)
        self.vit_3 = TransformerModel(dim=512, mlp_dim=2 * 512, heads=heads, depth=depth, sr_ratio=1)

        self.dul_vit_3 = DecoderBottleneckLayer(self.vit_3.dim)
        self.dul_cnn_4 = DecoderBottleneckLayer(self.vit_3.dim)
        self.dul_v3_cat_v2 = DecoderBottleneckLayer(2 * self.vit_2.dim)
        self.dul_e4_cat_e3 = DecoderBottleneckLayer(2 * self.vit_2.dim)
        self.dul_v2_cat_v1 = DecoderBottleneckLayer(self.vit_2.dim + self.vit_1.dim)
        self.dul_e3_cat_e2 = DecoderBottleneckLayer(self.vit_2.dim + self.vit_1.dim)

        self.up_vit_3 = nn.ConvTranspose2d(self.vit_3.dim, self.vit_2.dim, kernel_size=2, stride=2)
        self.up_cnn_4 = nn.ConvTranspose2d(self.vit_3.dim, self.vit_2.dim, kernel_size=2, stride=2)
        self.up_vit_3_2 = nn.ConvTranspose2d(self.vit_3.dim, self.vit_2.dim, kernel_size=2, stride=2)
        self.up_cnn_4_3 = nn.ConvTranspose2d(self.vit_3.dim, self.vit_2.dim, kernel_size=2, stride=2)

        self.se1 = SEBlock(1024)
        self.se2 = SEBlock(2 * 384)
        self.dul_cnn_cat_vit_1 = DecoderBottleneckLayer(2 * self.vit_3.dim)
        self.up_cnn_cat_vit_1 = nn.ConvTranspose2d(2 * self.vit_3.dim, self.vit_3.dim, kernel_size=2, stride=2)
        self.dul_cnn_cat_vit_3 = DecoderBottleneckLayer(1280)
        self.up_cnn_cat_vit_3 = nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=4)
        self.out = nn.Conv2d(640, num_classes, kernel_size=1)

    def forward(self, x):
        b, _, _, _ = x.shape
        x = x.repeat(1, 3, 1, 1)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.cnn_1(e0)  # 64     112
        e2 = self.cnn_2(e1)  # 128    56
        v1 = e2.permute(0, 2, 3, 1).contiguous()  # (b, 56, 56, 128)
        v1 = v1.view(b, -1, 128)
        v1, _ = self.vit_1(v1)
        v1 = v1.view(b, 56, 56, 128)
        v1 = v1.permute(0, 3, 1, 2).contiguous()   # 128, 56, 56

        e3 = self.cnn_3(v1)  #  28, 28, 256
        v2 = e3.permute(0, 2, 3, 1).contiguous()  # (b, 28, 28, 256)
        v2 = v2.view(b, -1, 256)
        v2, _ = self.vit_2(v2)
        v2 = v2.view(b, 28, 28, 256)
        v2 = v2.permute(0, 3, 1, 2).contiguous()  # 256, 28, 28

        e4 = self.cnn_4(v2)  # 14, 14, 512
        v3 = e4.permute(0, 2, 3, 1).contiguous()  # (b, 14, 14, 512)
        v3 = v3.view(b, -1, 512)
        v3, _ = self.vit_3(v3)
        v3 = v3.view(b, 14, 14, 512)
        v3 = v3.permute(0, 3, 1, 2).contiguous()  # 512, 14, 14

        v3 = self.dul_vit_3(v3)
        e4 = self.dul_cnn_4(e4)

        v3_up = self.up_vit_3(v3)  # 256, 28, 28
        e4_up = self.up_cnn_4(e4)  # 256, 28, 28
        v3_cat_v2 = torch.cat([v2, v3_up], dim=1)  # 512, 28, 28
        e4_cat_e3 = torch.cat([e3, e4_up], dim=1)  # 512, 28, 28
        v3_v2_dul = self.dul_v3_cat_v2(v3_cat_v2)  # 512, 28, 28 *****
        e4_e3_dul = self.dul_e4_cat_e3(e4_cat_e3)  # 512, 28, 28 *****

        v3_v2_up = self.up_vit_3_2(v3_v2_dul)     # 256, 56, 56
        e4_e3_up = self.up_cnn_4_3(e4_e3_dul)     # 256, 56, 56

        v2_cat_v1 = torch.cat([v3_v2_up, v1], dim=1)  # 256 + 128 ,56, 56
        e3_cat_e2 = torch.cat([e4_e3_up, e2], dim=1)  # 256 + 128 ,56, 56
        v2_v1_dul = self.dul_v2_cat_v1(v2_cat_v1)  # 256 + 128 ,56, 56  *****
        e3_e2_dul = self.dul_e3_cat_e2(e3_cat_e2)  # 256 + 128 ,56, 56  *****

        cnn_cat_vit_1 = torch.cat([v3_v2_dul, e4_e3_dul], dim=1)  # 2*512, 28, 28
        cnn_cat_vit_1 = self.se1(cnn_cat_vit_1)
        cnn_cat_vit_1 = self.dul_cnn_cat_vit_1(cnn_cat_vit_1)
        cnn_cat_vit_1_up = self.up_cnn_cat_vit_1(cnn_cat_vit_1)   # 512 , 56, 56

        cnn_cat_vit_2 = torch.cat([v2_v1_dul, e3_e2_dul], dim=1)  # 2*384, 56, 56
        cnn_cat_vit_2 = self.se2(cnn_cat_vit_2)                   # 2*384, 56, 56
        cnn_cat_vit_3 = torch.cat([cnn_cat_vit_2, cnn_cat_vit_1_up], dim=1)  # 512 + 2*384(1280), 56, 56
        cnn_cat_vit_3 = self.dul_cnn_cat_vit_3(cnn_cat_vit_3)                # 512 + 2*384(1280), 56, 56
        cnn_cat_vit_3_up = self.up_cnn_cat_vit_3(cnn_cat_vit_3)  # 640 , 224, 224

        out = self.out(cnn_cat_vit_3_up)  # num_classes , 224, 224

        return out

if __name__ == '__main__':
    a = torch.rand([2,3,224,224]).cuda()
    net = Cross_U_Like().cuda()
    print(net(a).shape)













