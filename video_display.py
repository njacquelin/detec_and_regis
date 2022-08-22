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

import time

from model import Unet_like, deeper_Unet_like, vanilla_Unet
from blobs_util2 import get_boxes_faster, a_link_to_the_past
from video_display_dataloader import get_video_dataloaders


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
    # size = (128, 128)
    size = (256, 256)
    # size = (512, 512)
    # size = (1024, 1024)

    display_bboxes = True
    display_heatmap = False
    text_addition = '_bboxes' if display_bboxes else ''
    text_addition += '_heatmap' if display_heatmap else ''

    heatmap_threshold = 0.45
    # centroid_threshold = 0.5
    apply_thresholds = True

    epochs = 200

    path = 'colorShifts_deeper_zoomedOut_'+str(epochs)+'epochs.pth'
    # path = 'no_sideSwitch_deeper_zoomedOut_'+str(epochs)+'epochs.pth'
    # path = 'deeper_zoomedOut_'+str(epochs)+'epochs.pth'
    # path = 'yes_homography_' + str(epochs) + 'epochs.pth'
    # path = 'yes_homography_BCE' + str(epochs) + 'epochs.pth'
    # path = "yes_130epochs.pth"
    # path = "vanilla"+str(epochs)+'epochs.pth'
    # path = str(epochs)+'epochs.pth'

    # model = vanilla_Unet()
    model = deeper_Unet_like()

    # models_path = './hardBox/'
    models_path = './models/'

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

    ################# DON'T TOUCH BEYOND THIS POINT ####################

    video_name = get_video_name(epochs, full_images_path, size, text_addition)
    video_path = './videos/' + video_name

    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()

    video_flow = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (size[1], size[0]))
    prev_boxes = []
    timer_limit = 10
    min_blob_size = 5

    dataloader = get_video_dataloaders(full_images_path, size, batch_size=4)

    img_nb = len(dataloader.dataset)


    i = 0
    j = 1
    start = time.time()
    with no_grad() :
        for batch in dataloader :
            batch_tensors = batch['tensor_img'].cuda()

            out_centroids = model(batch_tensors)

            out = out_centroids[:, 0]
            out = torch.unsqueeze(out, 1)
            # centroids = out_centroids[:, 1]
            # centroids = torch.unsqueeze(centroids, 1)

            out = tensor_to_image(out, False, batched=True)
            # centroids = tensor_to_image(centroids, False, batched=True)

            if apply_thresholds :
                out = np.where(out > heatmap_threshold, 1, 0)
                # centroids = np.where(centroids > centroid_threshold, 1, 0)

            batch_out = np.concatenate((out, out, out), axis=3)

            # img = cv2.addWeighted(img, 0.5, out, 0.5, 0)

            batch_img = batch['img']
            imgs = batch_img.numpy()

            for img, out in zip(imgs, batch_out) :

                if display_bboxes :
                    boxes = get_boxes_faster(out[:], threshold=heatmap_threshold, min_blob_size=min_blob_size)
                    img_overlay = img
                    for box in boxes:
                        index, prev_boxes = a_link_to_the_past(box, prev_boxes, IOU_threshold=0.2)
                        (xmin, ymin, xmax, ymax) = box
                        if index != -1:
                            empirical_max = 256
                            colour_nb_R = (333 ** index) % empirical_max
                            colour_nb_G = (222 ** index) % empirical_max
                            colour_nb_B = (111 ** index) % empirical_max
                            img = cv2.rectangle(img,
                                                (xmin, ymin),
                                                (xmax, ymax),
                                                (colour_nb_R, colour_nb_G, colour_nb_B),
                                                2)
                            # img = cv2.rectangle(img,
                            #                     (xmin, ymin),
                            #                     (xmax, ymax),
                            #                     (255, 0, 0),
                            #                     2)
                        else:
                            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 5)
                        img_overlay = img
                    prev_boxes = [(box, timer + 1) for (box, timer) in prev_boxes if timer < timer_limit]

                if display_heatmap : # display heatmap
                    # out = np.concatenate([out, out, out], axis=2)*255
                    out *= 255
                    out = out.astype(np.uint8)
                    img_overlay = cv2.addWeighted(img, 0.5, out, 0.5, 0)

                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)
                video_flow.write(img_overlay)
                # video_flow.write(out)

                if i==31 :
                    print(i*j)
                    i = -1
                    j+=1
                i += 1
    video_flow.release()

    end = time.time()
    duration = end - start
    print("FPS =", img_nb / duration)
