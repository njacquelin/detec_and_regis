import torch
from torch import load, unsqueeze, stack, no_grad
from torchvision import transforms
from torchvision.transforms.functional import rotate as rotate_tensor

import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
from scipy.ndimage import rotate
from cv2 import addWeighted
import cv2
import numpy as np
from matplotlib import pyplot as plt


from model import Unet_like, deeper_Unet_like, vanilla_Unet
from blobs_util2 import get_boxes_faster, a_link_to_the_past
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


def tensor_to_image(out, inv_trans=True, batched=False, to_uint8=True) :
    if batched : index_shift = 1
    else : index_shift = 0
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans :
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    if to_uint8 :
        out *= 255
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


def get_video_name(epochs, full_images_path, size, text_addition="") :
    video = full_images_path.split('/')[-1]
    # video = video[:-7] # remove "_frames" at the end of the name
    video_epochs = video + "_" + str(epochs)
    video_epochs_academy = video_epochs + '_' + str(size[0]) + text_addition
    video_epochs_avi = video_epochs_academy + '.avi'
    return video_epochs_avi


def get_detection_model(detection_path, models_path = './models/'):
    # model = vanilla_Unet()
    model = deeper_Unet_like()
    model_path = os.path.join(models_path, detection_path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()
    return model


def get_registration_model(path, models_path='./models/', field_length=50, field_width=25):
    markers_x = np.linspace(0, field_length, 11)
    lines_y = np.linspace(0, field_width, 11)
    model = vanilla_Unet(final_depth=len(markers_x) + len(lines_y))
    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()
    return model, field_width, field_length, markers_x, lines_y


if __name__=='__main__':
    # size = (128, 128)
    size = (256, 256)
    # size = (512, 512)
    # size = (1024, 1024)

    display_bboxes = True
    display_heatmap = False

    detection_heatmap_threshold = 0.45
    detection_model = get_detection_model(detection_path='colorShifts_deeper_zoomedOut_200epochs.pth')

    registration_threshold = 0.75
    registration_model, field_width, field_length, markers_x, lines_y = get_registration_model(path='100epochs_DECENT_all_train.pth')

    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/0 these case study'
    # full_images_path = './une_image_DONE'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/Gwangju_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/Angers19_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/Rennes19_frames'
    full_images_path = '/home/nicolas/swimmers_tracking/extractions/TITENIS_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/50_dos_dames_finaleA_f122020'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/50_brasse_dames_finaleA_f122020_droite'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/50_brasse_dames_finaleA_f122020_gauche'
    # full_images_path = '/home/amigo/Documents/Neptune/FFN_tool/2021_Nice_brasse_50_finaleA_dames/images/left'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/labelled_images/test'

    video_name = get_video_name(0, full_images_path, size)
    video_path = './videos/' + video_name
    video_flow = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (size[1], size[0]))

    dataloader = get_video_dataloaders(full_images_path, size, batch_size=4)

    img_nb = len(dataloader.dataset)

    i = 0
    j = 1
    with no_grad() :
        for batch in dataloader :
            batch_img = batch['img']
            imgs = batch_img.numpy()

            batch_tensors = batch['tensor_img'].cuda()

            ### DETECTION ###
            out_detection = detection_model(batch_tensors)
            out = out_detection[:, 0]
            out = torch.unsqueeze(out, 1)
            out = tensor_to_image(out, False, batched=True, to_uint8=False)
            out = np.where(out > detection_heatmap_threshold, 1, 0)
            batch_out = np.concatenate((out, out, out), axis=3)

            detected_imgs = []
            for img, out in zip(imgs, batch_out) :
                if display_bboxes :
                    boxes = get_boxes_faster(out[:], threshold=detection_heatmap_threshold)
                    img_overlay = img
                    for (xmin, ymin, xmax, ymax) in boxes:
                        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 5)
                        img_overlay = img
                if display_heatmap : # display heatmap
                    # out = np.concatenate([out, out, out], axis=2)*255
                    out *= 255
                    out = out.astype(np.uint8)
                    img_overlay = cv2.addWeighted(img, 0.5, out, 0.5, 0)

                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)
                detected_imgs.append(img_overlay)
            ##################

            ### REGISTRATION ###
            batch_out = registration_model(batch_tensors)
            batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)

            for img, out in zip(detected_imgs, batch_out):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(img, out, registration_threshold,
                                                                                  write_on_image=False,
                                                                                  lines_nb=len(lines_y),
                                                                                  markers_x=markers_x, lines_y=lines_y)
                src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
                H = get_homography_from_points(src_pts, dst_pts, size,
                                               field_length=field_length, field_width=field_width)
                if H is not None:
                    img = cv2.warpPerspective(img, H, size)
            ##################

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video_flow.write(img)

    video_flow.release()
