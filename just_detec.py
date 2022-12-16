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


if __name__=='__main__':
    # size = (128, 128)
    size = (256, 256)
    # size = (512, 512)
    # size = (1024, 1024)

    display_bboxes = False
    display_heatmap = True
    apply_thresholds = False
    detection_heatmap_threshold = 0.45

    model = get_detection_model(detection_path='colorShifts_deeper_zoomedOut_200epochs.pth')

    full_images_path = '/home/nicolas/swimmers_tracking/extractions/0 these case study'

    text_addition = ""
    text_addition += "_bboxes" if display_bboxes else ""
    text_addition += "_blobs" if display_heatmap and apply_thresholds \
                    else "_blobs_noThreshold" if not apply_thresholds \
                    else ""
    video_name = get_video_name(0, full_images_path, size, text_addition)
    video_path = './videos/' + video_name
    video_flow = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (size[1], size[0]))

    dataloader = get_video_dataloaders(full_images_path, size, batch_size=4)

    img_nb = len(dataloader.dataset)

    i = 0
    j = 1
    with no_grad() :
        for batch in dataloader :
            batch_tensors = batch['tensor_img'].cuda()

            out_centroids = model(batch_tensors)

            out = out_centroids[:, 0]
            out = torch.unsqueeze(out, 1)

            out = tensor_to_image(out, False, batched=True)

            if apply_thresholds :
                out = np.where(out > detection_heatmap_threshold, 1, 0)

            batch_out = np.concatenate((out, out, out), axis=3)

            batch_img = batch['img']
            imgs = batch_img.numpy()

            for img, out in zip(imgs, batch_out) :

                if display_bboxes :
                    boxes = get_boxes_faster(out[:], threshold=detection_heatmap_threshold)
                    img_overlay = img
                    for (xmin, ymin, xmax, ymax) in boxes:
                        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
                        img_overlay = img

                if display_heatmap : # display heatmap
                    # out = np.concatenate([out, out, out], axis=2)*255
                    out *= 255
                    out = out.astype(np.uint8)
                    img_overlay = cv2.addWeighted(img, 0.5, out, 0.5, 0)

                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)
                video_flow.write(img_overlay)

                if i==31 :
                    print(i*j)
                    i = -1
                    j+=1
                i += 1
    video_flow.release()
