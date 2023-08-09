from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet34
from torchcam.methods import SmoothGradCAMpp, GradCAMpp,XGradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import torch
import os
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
import cv2
from torchvision import transforms
# from model.network import U_Net
# from model.cross_u_trans_ori import cross_u_trans
from model.cross_u_trans_ori_nose import cross_u_trans
# from Image_Segmentation.FAT_net import FAT_Net
# from study.model import Model_4,Model_4_4
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

methods = 'nose'
# methods = 'ori'
# model = resnet18(pretrained=True).eval()
model = cross_u_trans(num_classes=9).eval()
weights_dict = torch.load(r"./train_ckpt/cross_u_trans_origin_nose_Synapse224_0.5ce_0.5dice_0.01base_lr_150/cross_u_trans_origin_nose_epo150_bs4_224/checkpoint_cross_u_trans_origin_nose_epoch_149.pth", map_location="cpu")
model.load_state_dict(weights_dict, strict=False)
# Get your input
# Load the NIfTI image
nifti = nib.load("./predictions/predictions_32/cross_u_trans_originSynapse224_0.5ce_0.5dice_0.01base_lr_150/cross_u_trans_origin_epo150_bs4_224/case0032_img.nii.gz")

# Get the image data as a numpy array
img_array = np.array(nifti.dataobj)

# Access a specific slice (e.g. slice 50 along axis 2 - axial plane)
slicer_num = 83
img = img_array[:, :, slicer_num]
plt.imshow(img)
plt.savefig('./output/feature_map_ori/out_83_4_nose.jpg')


# Preprocess it for your chosen model

img = Image.fromarray(img)
img = transforms.functional.resize(img, (224, 224))
img = np.array(img)
img = torch.from_numpy(img).repeat(3, 1, 1)
# img_tensor = torch.stack([img_tensor[0]]*3, dim=0)
# print(model)
block = 'out'
cam_extractor = XGradCAM(model, block)
# Preprocess your data and feed it to the model
out = model(img.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
class_idx = 4
activation_map = cam_extractor(class_idx=class_idx, scores=out)
# Resize the CAM and overlay it
cam_pil = to_pil_image(activation_map[0].squeeze(0), mode='F')
result = overlay_mask(to_pil_image(img),cam_pil, alpha=0.4)
# Display it
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.savefig('./output/no_se/{}_{}_{}_classidx{}.png'.format(methods,slicer_num, block, class_idx))