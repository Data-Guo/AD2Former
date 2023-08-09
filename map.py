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
from scipy.ndimage import zoom
# from model.network import U_Net
from model.cross_u_trans_ori import cross_u_trans
# from Image_Segmentation.FAT_net import FAT_Net
# from study.model import Model_4,Model_4_4
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


# model = resnet18(pretrained=True).eval()
model = cross_u_trans(num_classes=9).eval()
weights_dict = torch.load(r"./train_ckpt/cross_u_trans_originSynapse224_0.5ce_0.5dice_0.01base_lr_150/cross_u_trans_origin_epo150_bs4_224/checkpoint_cross_u_trans_origin_epoch_149.pth", map_location="cpu")
model.load_state_dict(weights_dict, strict=False)
# Get your input
nifti = nib.load("./predictions/predictions_32/cross_u_trans_originSynapse224_0.5ce_0.5dice_0.01base_lr_150/cross_u_trans_origin_epo150_bs4_224/case0032_pred.nii.gz")

# Get the image data as a numpy array
img_array = np.array(nifti.dataobj)

# Access a specific slice (e.g. slice 50 along axis 2 - axial plane)
img = img_array[:, :, 130]
# Preprocess it for your chosen model
resized_img = zoom(img, (224/img.shape[0], 224/img.shape[1]))
# print(resized_img.shape)#224 224
tensor_img = torch.from_numpy(resized_img / 255.).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 2, 3)
tensor_img  =  tensor_img.repeat(1, 3, 1, 1)
input_tensor = normalize(tensor_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# input_tensor = normalize(torch.from_numpy(zoom(img, (224/img.shape[0], 224/img.shape[1])) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
# input_tensor = normalize(resize(img, (224, 224)) / 255., [0, 0, 0], [1, 1, 1])
cam_extractor = XGradCAM(model, "up_cnn_cat_vit_3")
# Preprocess your data and feed it to the model
print(input_tensor.shape)
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
# activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
activation_map = cam_extractor(class_idx=8, scores=out)
# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.8)
# Display it
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.savefig('./output/up_cnn_cat_vit_3.png')
# plt.show()