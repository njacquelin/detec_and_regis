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
import matplotlib
matplotlib.use('TkAgg')

from scipy.signal import medfilt

from model import deeper_Unet_like, vanilla_Unet

from video_display_dataloader import get_video_dataloaders
from grid_utils import get_landmarks_positions, get_faster_landmarks_positions,\
     get_homography_from_points, conflicts_managements, display_on_image

from just_regis import tensor_to_image, get_video_name, get_registration_model


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def smooth_homographies(homographies, k_size=51):
    kernel_size = (k_size, 1, 1)
    mid_kernel = (k_size - 1) // 2
    homographies = medfilt(np.array(homographies), kernel_size=kernel_size)
    for x in range(3):
        for y in range(3):
            homographies[mid_kernel:-mid_kernel, x, y] = running_mean(homographies[:, x, y], N=kernel_size[0])
    return homographies, mid_kernel


if __name__=='__main__':
    size = (256, 256)
    registration_threshold = 0.75
    registration_model, field_width, field_length, markers_x, lines_y = get_registration_model(path='100epochs_DECENT_all_train.pth')

    batch_size = 4
    k_size = 51

    # if False, warp template
    warp_image = True
    smooth_homography = True

    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/TITENIS_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/0 these case study'
    full_images_path = "../datasets/race_example/2021_Nice_freestyle_50_serie4_hommes_fixeGauche.mp4"

    text_addition = "_smoothed_homography" if smooth_homography else "_homography"
    text_addition += "_warped" if warp_image else ""
    video_name = get_video_name(0, full_images_path, size, text_addition, True, registration_threshold)
    video_path = './videos/' + video_name
    video_flow = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, size[::-1])

    dataloader = get_video_dataloaders(full_images_path, size, batch_size=batch_size)

    template = cv2.imread("../pool_template.jpg")
    template = cv2.resize(template, size)

    homographies = []
    i = 0
    j = 1
    with no_grad() :
        print('Computing matrices...')
        for batch in dataloader :
            batch_tensors = batch['tensor_img'].cuda()

            batch_out = registration_model(batch_tensors)
            batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)

            batch_img = batch['img']
            imgs = batch_img.numpy()

            for img, out in zip(imgs, batch_out) :
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(img, out, registration_threshold,
                                                                                  write_on_image=False,
                                                                                  lines_nb=len(lines_y),
                                                                                  markers_x=markers_x, lines_y=lines_y)
                src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
                H = get_homography_from_points(src_pts, dst_pts, size,
                                               field_length=field_length, field_width=field_width)
                homographies.append(H)

                if i==31 :
                    print(i*j)
                    i = -1
                    j+=1
                i += 1
    print("Matrices computed.")

    if smooth_homography:
        print("Smoothing...")
        homographies, mid_kernel = smooth_homographies(homographies, k_size=k_size)
        print("Smoothing finished.")
    else:
        mid_kernel = (k_size-1) // 2

    print("Registration started...")
    for root, dirs, files in os.walk(full_images_path):
        files.sort()
        files.sort(key=len)
        for H, file in zip(homographies[mid_kernel:-mid_kernel], files[mid_kernel:-mid_kernel]):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)

            if warp_image:
                if H is not None:
                    img = cv2.warpPerspective(img, H, size)
                img = cv2.addWeighted(img, 0.5, template, 0.5, 0)
            else:
                if H is not None and np.linalg.det(H) != 0:
                    template2 = cv2.warpPerspective(template, np.linalg.inv(H), size)
                else: template2 = template[:]
                img = cv2.addWeighted(img, 0.5, template2, 0.5, 0)



            video_flow.write(img)
    print("Registration Finished. Video Ready. Smoothed with a window of size " + str(k_size) + ".")
    video_flow.release()
