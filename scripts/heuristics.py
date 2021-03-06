# Imports
import os
import cv2
import json
import torch
import numpy as np


# Bounding Box filter class
class BBox:
    def __init__(self, inputs, bbox_path):
        # Inputs are paths (do not support tensors)
        if isinstance(inputs, np.ndarray) is False:
            raise NotImplementedError('Does not support tensor inputs')
        self.paths = inputs
        self.bbox_path = bbox_path

    # Get bbox for frame
    def get_frame_bbox(self, frame_path):
        key_name = os.path.basename(frame_path)
        file_name = key_name[:-10] + '.json'
        file_path = os.path.join(self.bbox_path, file_name)
        with open(file_path) as json_file:
            json_data = json.load(json_file)
        bbox = json_data[key_name]['object_bounding_box']
        bbox = torch.tensor(bbox)
        return bbox

    # Get bboxes for clip
    def get_clip_bbox(self, clip_path):
        clip_bbox_list = list()
        for frame_path in clip_path:
            frame_bbox = self.get_frame_bbox(frame_path)
            clip_bbox_list.append(frame_bbox)
        clip_bbox = torch.stack(clip_bbox_list)
        return clip_bbox
    
    # Get bboxes for batch
    def get_batch_bbox(self):
        batch_bbox_list = list()
        for clip_path in self.paths:
            clip_bbox = self.get_clip_bbox(clip_path)
            batch_bbox_list.append(clip_bbox)
        batch_bbox = torch.stack(batch_bbox_list)
        return batch_bbox
    
    # Rank bboxes from biggest to smallest
    def get_ranked_bbox_sizes(self):
        batch_bbox = self.get_batch_bbox()
        batch_bbox_list = list()
        for clip_bbox in batch_bbox:
            clip_bbox_list = list()
            for frame_bbox in clip_bbox:
                frame_bbox_size = frame_bbox[2] * frame_bbox[3]
                clip_bbox_list.append(frame_bbox_size)
            clip_bbox_size = sum(clip_bbox_list) / len(clip_bbox_list)
            batch_bbox_list.append(clip_bbox_size)
        sorted_idxs = sorted(range(len(batch_bbox_list)), key=lambda i: batch_bbox_list[i], reverse=True)
        return sorted_idxs
        

# Blur filter class
class Blur:
    def __init__(self, inputs):
        if torch.is_tensor(inputs):
            self.images = inputs
        else:
            raise NotImplementedError('Does not support path inputs')
    
    # Compute blur of image tensor
    def compute_img_blur(self, tensor_img):
        # Detach from GPU if on cuda device using
        # .cpu().detach().numpy()
        numpy_img = tensor_img.numpy()
        # Make channel dimension (currently 0) as 2
        numpy_img = np.transpose(numpy_img, (2, 1, 0))
        gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)
        # Convert dtype to np.float64 or higher precision
        gray_img = gray_img.astype(np.float64)
        # Low blur score corresponds to more blur
        # So we need to keep images with higher scores
        blur_score = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        return blur_score

    # Compute blue of clip tensor
    def compute_clip_blur(self, tensor_clip):
        clip_blur_list = list()
        for tensor_img in tensor_clip:
            img_blur_score = self.compute_img_blur(tensor_img)
            clip_blur_list.append(img_blur_score)
        clip_blur = sum(clip_blur_list) / len(clip_blur_list)
        return clip_blur
    
    # Returns idxs of clips in batch by increasing order of blur score
    # Therefore, first k clips in batch will be most blurry
    def get_ranked_blur_scores(self):
        input_blur_list = list()
        for tensor_clip in self.images:
            clip_blur_score = self.compute_clip_blur(tensor_clip)
            input_blur_list.append(clip_blur_score)
        sorted_idxs = sorted(range(len(input_blur_list)), key=lambda i: input_blur_list[i], reverse=False)
        return sorted_idxs
