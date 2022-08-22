import torch
from torch import load, unsqueeze, no_grad
from torchvision import transforms

import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
from scipy.ndimage import rotate
from cv2 import addWeighted
import cv2
import numpy as np
from matplotlib import pyplot as plt

from model import Unet_like, deeper_Unet_like, vanilla_Unet
from video_display_dataloader import get_video_dataloaders

from grid_utils import get_faster_landmarks_positions,\
     get_homography_from_points, conflicts_managements


def compare(out, img, thresholod=None):
    heatmap = np.absolute(out - img)
    if thresholod is not None :
        heatmap = np.where(heatmap > thresholod, 1., 0.)
    heatmap = np.amax(heatmap, 2)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    return heatmap


def tensor_to_image(out, inv_trans=True, batched=False) :
    if batched : index_shift = 1
    else : index_shift = 0
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans :
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    out = out.astype(np.float64)
    out = np.swapaxes(out, index_shift + 0, index_shift + 2)
    out = np.swapaxes(out, index_shift + 0, index_shift + 1)
    return out

def tensor_to_image2(out, inv_trans=True, batched=False, to_uint8=True) :
    if batched : index_shift = 1
    else : index_shift = 0
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans :
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    if to_uint8 :
        out *= 256
        out = out.astype(np.uint8)
    out = np.swapaxes(out, index_shift + 0, index_shift + 2)
    out = np.swapaxes(out, index_shift + 0, index_shift + 1)
    return out


def get_transform(x) :
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    tensor = img_transform(x)
    tensor = unsqueeze(tensor, 0).float()
    return tensor.cuda()


def get_video_name(epochs, full_images_path, size, text_addition) :
    video = full_images_path.split('/')[-1]
    # video = video[:-7] # remove "_frames" at the end of the name
    video_epochs = video + "_" + str(epochs)
    video_epochs_academy = video_epochs + '_' + str(size[0]) + text_addition
    video_epochs_avi = video_epochs_academy + '.avi'
    return video_epochs_avi


if __name__=='__main__':
    size = (256, 256)
    full_images_path = './une_image_TODO'
    out_img_dir = './une_image_DONE/'
    dataloader = get_video_dataloaders(full_images_path, size, batch_size=1)
    img_nb = len(dataloader.dataset)

    ### DETEC PREPROC ###
    heatmap_threshold = 0.45
    apply_thresholds = False
    epochs = 200
    path = 'colorShifts_deeper_zoomedOut_200epochs.pth'
    model = deeper_Unet_like()
    models_path = './models/'

    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()
    ######################

    ### REGIS PREPROC ###
    threshold = 0.75
    field_length = 115
    markers_x = np.linspace(0, field_length, 11)
    field_width = 74
    lines_y = np.linspace(0, field_width, 11)
    regis_path = '100epochs_DECENT_all_train.pth'
    regis_model = vanilla_Unet(final_depth=len(markers_x) + len(lines_y))
    regis_models_path = './models/'

    regis_models_path = os.path.join(regis_models_path, regis_path)
    regis_model.load_state_dict(load(regis_models_path))
    regis_model = regis_model.cuda()
    regis_model.eval()
    #####################

    counter = 0
    with no_grad() :
        for batch in dataloader :
            batch_img = batch['img']
            imgs = batch_img.numpy()[0]
            name = batch['name'][0]

            batch_tensors = batch['tensor_img'].cuda()

            ### DETECTION ###
            out_centroids = model(batch_tensors)
            out = out_centroids[:, 0]
            out = torch.unsqueeze(out, 1)
            out = tensor_to_image(out, False, batched=True)
            if apply_thresholds :
                out = np.where(out > heatmap_threshold, 1, 0)
            batch_out = np.concatenate((out, out, out), axis=3)[0]

            # for img, out in zip(imgs, batch_out) :
            #     out *= 255
            #     out = out.astype(np.uint8)
            #     img_overlay = cv2.addWeighted(img, 0.3, out, 0.9, 0)
            #     img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)
            batch_out *= 255
            batch_out = batch_out.astype(np.uint8)
            img_overlay = cv2.addWeighted(imgs, 0.5, batch_out, 0.5, 0)
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)
            ###################

            ### REGISTRATION ###
            batch_out = regis_model(batch_tensors)
            batch_out = tensor_to_image2(batch_out, inv_trans=False, batched=True, to_uint8=False)
            batch_out = batch_out[0]

            # for img, out in zip(imgs, batch_out):
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #
            #     img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(img, out, threshold,
            #                                                                       write_on_image=True,
            #                                                                       lines_nb=len(lines_y),
            #                                                                       markers_x=markers_x,
            #                                                                       lines_y=lines_y)
            #     src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
            #     H = get_homography_from_points(src_pts, dst_pts, size,
            #                                        field_length=field_length, field_width=field_width)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            imgs, src_pts, dst_pts, entropies = get_faster_landmarks_positions(imgs, batch_out, threshold,
                                                                              write_on_image=True,
                                                                              lines_nb=len(lines_y),
                                                                              markers_x=markers_x,
                                                                              lines_y=lines_y)
            src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
            H = get_homography_from_points(src_pts, dst_pts, size,
                                               field_length=field_length, field_width=field_width)
            #####################

            img_overlay = cv2.warpPerspective(img_overlay, H, size)
            img_overlay = cv2.resize(img_overlay, (512, 256))
            out_img_path = out_img_dir + name + ".jpg"
            cv2.imwrite(out_img_path, img_overlay)
            counter += 1
