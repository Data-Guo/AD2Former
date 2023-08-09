import torch
from torchvision import models as resnet_model
from torch import nn
from model.transformerv2 import TransformerModel

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

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)   # B H/2*W/2 2*C

        return x

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

class PyramidFormer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, mlp_dim, depth, heads, pool=False):
        super(PyramidFormer, self).__init__()
        self.pool = pool
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        if pool is False:
            self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.vit = TransformerModel(dim=embed_dim, mlp_dim=mlp_dim,depth=depth, heads=heads)
        else:
            self.pooling = nn.AdaptiveAvgPool2d((28, 28))
            self.vit = TransformerModel(dim=in_channels, mlp_dim=mlp_dim, depth=depth, heads=heads)


    def forward(self, x):
        b = x.shape[0]
        size = x.shape[2:]
        h, w = x.shape[2], x.shape[3]
        pool = self.pool
        embed_dim = self.embed_dim
        patch_size = self.patch_size
        self.sample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
        if pool is False:
            out, _ = self.vit(self.patch_embed(x).permute(0, 2, 3, 1).contiguous().view(b, -1, embed_dim))
            out = out.view(b, int(h / patch_size), int(w / patch_size), embed_dim).permute(0, 3, 1, 2).contiguous()
        else:
            out, _ = self.vit(self.pooling(x).permute(0, 2, 3, 1).contiguous().view(b, -1, embed_dim))
            out = out.view(b, int(h / patch_size), int(w / patch_size), embed_dim).permute(0, 3, 1, 2).contiguous()
        # output = nn.Upsample(out, size, mode='bilinear')
        return self.sample(out)

class DeepTR(nn.Module):
    def __init__(self, n_channels=3, num_classes=9, heads=8, depth=3, dim=320, patch_size=[14, 8, 4, 2]):
        super(DeepTR, self).__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.heads = heads
        self.depth = depth
        self.dim = dim
        resnet = resnet_model.resnet34(pretrained=False)  #weights=resnet_model.ResNet34_Weights.DEFAULT
        self.firstconv = resnet.conv1  # 224x224
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1  # 64    ,112
        self.encoder2 = resnet.layer2  # 128   ,56
        self.MultiScale_1 = PyramidFormer(in_channels = 128, embed_dim = dim, patch_size = patch_size[0], mlp_dim = 2*patch_size[0], depth = depth, heads = heads, pool=False)
        self.MultiScale_2 = PyramidFormer(in_channels=128, embed_dim=dim, patch_size=patch_size[1],
                                          mlp_dim=2 * patch_size[1], depth=depth, heads=heads, pool=False)
        self.MultiScale_3 = PyramidFormer(in_channels=128, embed_dim=dim, patch_size=patch_size[2],
                                          mlp_dim=2 * patch_size[2], depth=depth, heads=heads, pool=False)
        self.MultiScale_4 = PyramidFormer(in_channels=128, embed_dim=dim, patch_size=patch_size[3],
                                          mlp_dim=2 * patch_size[3], depth=depth, heads=heads, pool=False)
        self.MultiScale_5 = PyramidFormer(in_channels=128, embed_dim=128, patch_size=2, mlp_dim=2 * 128, depth=depth, heads=heads, pool=True)
        self.project = nn.Conv2d(4*dim + 128, 512, kernel_size=1)
        self.cat_se = SEBlock(768)
        self.low_se = SEBlock(256)
        self.decoder1 = DecoderBottleneckLayer(768)
        self.up = nn.ConvTranspose2d(768, 64, kernel_size=4, stride=4)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        self.Merging_1 = PatchMerging((112, 112), 64)


    def forward(self,x):
        x = x.repeat(1, 3, 1, 1)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)  # 64     112
        e2 = self.encoder2(e1)  # 128    56

        M_1 = self.MultiScale_1(e2)  # 4x4 -> 56x56, dim
        M_2 = self.MultiScale_2(e2)  # 7x7 -> 56x56, dim
        M_3 = self.MultiScale_3(e2)  # 14x14 -> 56x56, dim
        M_4 = self.MultiScale_4(e2)  # 28x28 -> 56x56, dim
        M_5 = self.MultiScale_5(e2)  # 28x28 -> 56x56, 128
        M_all = torch.cat([M_1, M_2, M_3, M_4, M_5], dim=1)
        M_out = self.project(M_all)  # 56x56,512

        e1_Merging = e1.flatten(2).transpose(1, 2)  # b, 112*112, 64
        e1_Merging = self.Merging_1(e1_Merging)     # b ,56x56 ,128
        e1_Merging = e1_Merging.transpose(1, 2).view(x.shape[0], 128, 56, 56)  # b ,128,56,56

        low_feature = torch.cat([e2, e1_Merging], dim=1)   # b, 256, 56, 56
        low_feature_se = self.low_se(low_feature)  # b, 256, 56, 56

        low_cat = torch.cat([low_feature_se, M_out], dim=1)   # 56x56,768
        # low_cat_se = self.cat_se(low_cat)   # 56x56,768
        low_cat_se = self.decoder1(low_cat)  # 56x56,768
        out = self.up(low_cat_se)   # 224x224,64
        out = self.out(out)

        return out

if __name__ == '__main__':
    a = torch.rand([2,3,224,224]).cuda()
    net = DeepTR().cuda()
    print(net(a).shape)




