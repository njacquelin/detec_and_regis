import pickle

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


from blobs_util2 import get_boxes_faster, a_link_to_the_past
from video_display_dataloader import get_video_dataloaders
from just_detec import tensor_to_image, get_video_name, get_detection_model


def get_bbox(box, crop_size=(70, 50)):
    x_min, y_min, x_max, y_max = box

    if (x_max - x_min) / (y_max - y_min) < crop_size[1] / crop_size[0]:
        width = (y_max - y_min) * crop_size[1] / crop_size[0]
        center = (x_max + x_min) // 2
        x_min = int(center - width // 2)
        x_max = int(center + width // 2)
    else:
        heigh = (x_max - x_min) / (crop_size[1] / crop_size[0])
        center = (y_max + y_min) // 2
        y_min = int(center - heigh // 2)
        y_max = int(center + heigh // 2)

    bbox = x_min, x_max, y_min, y_max
    return bbox


def get_new_index(matched_boxes):
    if matched_boxes == {}: return 0
    keys = list(matched_boxes.keys())
    keys = [int(k) for k in keys]
    new_key = max(keys) + 1
    return new_key


def add_match(matched_boxes, index, box):
    if index == -1:
        new_box_index = get_new_index(matched_boxes)
        matched_boxes[new_box_index] = box
    else:
        matched_boxes[index] = box
    return matched_boxes


def reset_matches(matched_boxes):
    for k in matched_boxes.keys():
        matched_boxes[k] = None
    return matched_boxes


def find_crops_to_keep(boxes_through_time, keep_threshold = 0.9):
    video_length = len(boxes_through_time)
    retained_crops = []
    print("Retained crops :")
    for k in boxes_through_time[-1].keys():
        first_apparition = video_length
        last_apparition = 0
        total_apparitions = 0
        for i in range(len(boxes_through_time)):
            if len(boxes_through_time[i]) - 1 >= k and first_apparition == video_length:
                first_apparition = i
            if len(boxes_through_time[i]) - 1 >= k:
                if boxes_through_time[i][k] is not None:
                    total_apparitions += 1
                    last_apparition = i
        if total_apparitions / video_length > keep_threshold:
            print(k, round(total_apparitions / video_length, 2))
            retained_crops.append((k, first_apparition, last_apparition))
    return retained_crops


def crop_extraction(full_images_path, retained_crops, crop_size, tracking_data_path, video_name):
    video_flow = []
    for r, _, __ in retained_crops:
        video_path = os.path.join(tracking_data_path, video_name + "_" + str(r) + ".avi")
        video_flow.append(cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, crop_size))

    print("Extraction started...")
    i, j = 0, 0
    for root, dirs, files in os.walk(full_images_path):
        files.sort()
        files.sort(key=len)
        for file_nb, (boxes, file) in enumerate(zip(boxes_through_time, files)):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)

            for flow, (c, first_apparition, last_apparition) in zip(video_flow, retained_crops):
                if file_nb < first_apparition: continue
                if file_nb >= last_apparition: continue
                frame = np.ones((*crop_size[::-1], 3), dtype=np.uint8) * 128
                if c in boxes.keys() and boxes[c] is not None:
                    bbox = boxes[c]
                    bbox = get_bbox(bbox, crop_size)
                    if bbox[0] >= 0 and bbox[1] < 256 and \
                            bbox[2] >= 0 and bbox[3] <= 256:
                        frame = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
                        frame = cv2.resize(frame, crop_size)
                flow.write(frame)

            if i == 31:
                print(i * j)
                i = -1
                j += 1
            i += 1

    for flow in video_flow:
        flow.release()
    print("Extraction complete.")


if __name__=='__main__':
    size = (256, 256)
    min_box_size = 10 * 5
    crop_size = (40, 30)

    detection_heatmap_threshold = 0.45
    model_path = "colorShifts_deeper_zoomedOut_200epochs.pth"

    full_images_path = '/home/nicolas/swimmers_tracking/extractions/Gwangju_frames'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/TITENIS_frames_resized_256'
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/0 these case study'

    text_addition = "_tracking"
    video_name = get_video_name(0, full_images_path, size, text_addition)
    # video_path = './videos/' + video_name
    # video_flow = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (size[1], size[0]))

    tracking_data_path = "./tracking/" + video_name[:-4] # -4 to remove video extension
    tracking_data_file = os.path.join(tracking_data_path, "tracking_data.pkl")
    if not os.path.exists(tracking_data_path): os.mkdir(tracking_data_path)
    if not os.path.isfile(tracking_data_file):
        model = get_detection_model(detection_path=model_path)
        dataloader = get_video_dataloaders(full_images_path, size, batch_size=4)

        img_nb = len(dataloader.dataset)

        boxes_through_time = []
        matched_boxes = {}
        prev_boxes = []
        timer_limit = np.inf

        with no_grad() :
            i = 0
            j = 1
            for batch in dataloader :
                batch_tensors = batch['tensor_img'].cuda()

                out_centroids = model(batch_tensors)

                out = out_centroids[:, 0]
                out = torch.unsqueeze(out, 1)

                out = tensor_to_image(out, False, batched=True)
                out = np.where(out > detection_heatmap_threshold, 1, 0)

                batch_out = np.concatenate((out, out, out), axis=3)

                batch_img = batch['img']
                imgs = batch_img.numpy()

                for img, out in zip(imgs, batch_out) :
                    boxes = get_boxes_faster(out[:], threshold=detection_heatmap_threshold)
                    matched_boxes = reset_matches(matched_boxes)
                    for box in boxes:
                        (xmin, ymin, xmax, ymax) = box
                        if (xmax - xmin) * (ymax - ymin) < min_box_size: continue
                        index, prev_boxes = a_link_to_the_past(box, prev_boxes, IOU_threshold=0.2)
                        matched_boxes = add_match(matched_boxes, index, box)
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
                        else:
                            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 5)
                    prev_boxes = [(box, timer + 1) for (box, timer) in prev_boxes
                                  if timer < timer_limit
                                  and abs(box[2] - box[0]) * abs(box[1] - box[3]) > min_box_size]
                    boxes_through_time.append(matched_boxes.copy())

                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # video_flow.write(img)

                    if i==31 :
                        print(i*j)
                        i = -1
                        j+=1
                    i += 1
        # video_flow.release()
        with open(tracking_data_file, 'wb') as file:
            pickle.dump(boxes_through_time, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(tracking_data_file, 'rb') as file:
            boxes_through_time = pickle.load(file)

    retained_crops = find_crops_to_keep(boxes_through_time, keep_threshold=0.2)
    crop_extraction(full_images_path, retained_crops, crop_size, tracking_data_path, video_name)