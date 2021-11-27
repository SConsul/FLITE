# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.transforms import orbit_transform

class DatasetFromClipPaths(Dataset):
    def __init__(self, clip_paths, with_labels):
        super().__init__()
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        self.transformation = orbit_transform

    def __getitem__(self, index):
        clip = []
        for frame_path in self.clip_paths[index]:
            clip.append(self.transformation(frame_path))
        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index]
        else:
            return torch.stack(clip, dim=0)

    def __len__(self):
        return len(self.clip_paths)

class DatasetFromClipPathsBBox(Dataset):
    def __init__(self, clip_paths, bbox_path, with_labels):
        super().__init__()
        self.bbox_path = bbox_path
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        self.transformation = orbit_transform

    def __getitem__(self, index):
        clip = []
        bbox_list = []
        for frame_path in self.clip_paths[index]:
            clip.append(self.transformation(frame_path))

            key_name = os.path.basename(frame_path)
            file_name = key_name[:-10] + '.json'
            file_path = os.path.join(self.bbox_path, file_name)
            with open(file_path) as json_file:
                json_data = json.load(json_file)
            x, y, w, h = json_data[key_name]['object_bounding_box']
            bbox = torch.tensor([float(x)*224.0/1080.0, float(y)*224.0/1080.0, float(w)*224.0/1080.0, float(h)*224.0/1080.0])
            bbox_list.append(bbox)

        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index], torch.stack(bbox_list, dim=0)
        else:
            return torch.stack(clip, dim=0), torch.stack(bbox_list, dim=0)

    def __len__(self):
        return len(self.clip_paths)

class DatasetFromClipPathsBBoxAttn(Dataset):
    def __init__(self, clip_paths, bbox_path, with_labels):
        super().__init__()
        self.bbox_path = bbox_path
        self.with_labels = with_labels
        if self.with_labels:
            self.clip_paths, self.clip_labels = clip_paths
        else:
            self.clip_paths = clip_paths
        
        self.transformation = orbit_transform

    def __getitem__(self, index):
        clip = []
        bbox_list = []
        for frame_path in self.clip_paths[index]:
            key_name = os.path.basename(frame_path)
            file_name = key_name[:-10] + '.json'
            file_path = os.path.join(self.bbox_path, file_name)
            with open(file_path) as json_file:
                json_data = json.load(json_file)
            x, y, w, h = json_data[key_name]['object_bounding_box']
            bbox = torch.tensor([float(x)*224.0/1080.0, float(y)*224.0/1080.0, float(w)*224.0/1080.0, float(h)*224.0/1080.0])
            bbox_list.append(bbox)

            clip_frame = self.transformation(frame_path)
            x_center = bbox[0]
            y_center = bbox[1]
            width = bbox[2]
            height = bbox[3]
            x_left = int(x_center - width // 2)
            x_right = int(x_center + width // 2)
            y_up = int(y_center - height // 2)
            y_down = int(y_center + height // 2)
            clip_frame[:y_up, :] = torch.zeros(clip_frame[:y_up, :].shape)
            clip_frame[y_down:, :] = torch.zeros(clip_frame[y_down:, :].shape)
            clip_frame[:, :x_left] = torch.zeros(clip_frame[:, :x_left].shape)
            clip_frame[:, x_right:] = torch.zeros(clip_frame[:, x_right:].shape)

            clip.append(clip_frame)

            
        if self.with_labels:
            return torch.stack(clip, dim=0), self.clip_labels[index], torch.stack(bbox_list, dim=0)
        else:
            return torch.stack(clip, dim=0), torch.stack(bbox_list, dim=0)

    def __len__(self):
        return len(self.clip_paths)

def get_clip_loader(clips, batch_size, with_labels=False, with_bbox=False, bbox_path="", with_attn=False):
    if isinstance(clips[0], np.ndarray) and not with_bbox:
        clips_dataset = DatasetFromClipPaths(clips, with_labels=with_labels)
        return DataLoader(clips_dataset,
                      batch_size=batch_size,
                      num_workers=8,
                      pin_memory=True,
                      prefetch_factor=8,
                      persistent_workers=True)

    elif isinstance(clips[0], np.ndarray) and with_bbox:
        if with_attn:
            clips_dataset_with_bbox = DatasetFromClipPathsBBoxAttn(clips, bbox_path, with_labels=with_labels)
        else:
            clips_dataset_with_bbox = DatasetFromClipPathsBBox(clips, bbox_path, with_labels=with_labels)
        return DataLoader(clips_dataset_with_bbox,
                      batch_size=batch_size,
                      num_workers=8,
                      pin_memory=True,
                      prefetch_factor=8,
                      persistent_workers=True)
    elif isinstance(clips[0], torch.Tensor):
        if with_labels:
            return list(zip(clips[0].split(batch_size), clips[1].split(batch_size)))
        else: 
            return clips.split(batch_size)

def attach_frame_history(clips, labels, clip_length):
    
    # expand labels
    labels = labels.view(-1,1).repeat(1, clip_length).view(-1)

    if isinstance(clips, np.ndarray):
        return attach_frame_history_paths(clips, clip_length), labels
    elif isinstance(clips, torch.Tensor):
        return attach_frame_history_tensor(clips, clip_length), labels

def attach_frame_history_paths(clip_paths, clip_length):
    """
    Function to attach the immediate history of clip_length frames to each frame in an array of frame paths.
    :param clip_paths: (np.ndarray) Clip paths.
    :param clip_length: (int) Number of frames of history to append to each frame.
    :return: (np.ndarray) Clip paths with attached frame history.
    """
    # pad with first frame so that frames 0 to clip_length-1 can be evaluated
    frame_paths = clip_paths.reshape(-1)
    frame_paths = np.concatenate([np.repeat(frame_paths[0], clip_length-1), frame_paths])
    
    # for each frame path, attach its immediate history of clip_length frames
    frame_paths = [ frame_paths ]
    for l in range(1, clip_length):
        frame_paths.append( np.roll(frame_paths[0], shift=-l, axis=0) )
    frame_paths_with_history = np.stack(frame_paths, axis=1) # of size num_clips x clip_length
    
    # since frame_paths_with_history have wrapped around, remove last (clip_length - 1) frames
    return frame_paths_with_history[:-(clip_length-1)]

def attach_frame_history_tensor(clip_data, clip_length):
    """
    Function to attach the immediate history of self.clip_length frames to each frame in a tensor of frame data.
    param clip_data: (torch.Tensor) Frame data organised in clips of self.clip_length contiguous frames.
    :return: (torch.Tensor) Clip data with attached frame history.
    """
    # pad with first frame so that frames 0 to clip_length-1 can be evaluated
    clip_data = clip_data.flatten(end_dim=1)
    frame_0 = clip_data.narrow(0, 0, 1)
    clip_data = torch.cat((frame_0.repeat(clip_length-1, 1, 1, 1), clip_data), dim=0)

    # for each frame, attach its immediate history of clip_length frames
    clip_data = [ clip_data ]
    for l in range(1, clip_length):
        clip_data.append( clip_data[0].roll(shifts=-l, dims=0) )
    clip_data = torch.stack(clip_data, dim=1) # of size num_clips x clip_length
    
    # since clip_data has wrapped around, remove last (clip_length - 1) frames
    return clip_data[:-(clip_length-1)]

def unpack_task(task_dict, device, context_to_device=True, target_to_device=False, preload_clips=False):
    context_clips = task_dict['context_clips']
    context_clip_paths = task_dict['context_clip_paths']
    context_labels = task_dict['context_labels']
    target_clips = task_dict['target_clips']
    target_clip_paths = task_dict['target_clip_paths']
    target_labels = task_dict['target_labels']

    if context_to_device and isinstance(context_labels, torch.Tensor):
        context_labels = context_labels.to(device)
    if target_to_device and isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.to(device)
   
    if preload_clips:
        return context_clips, context_labels, target_clips, target_labels
    else:
        return context_clip_paths, context_labels, target_clip_paths, target_labels
