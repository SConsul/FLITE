# Imports
import cv2
import torch


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
        gray_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2GRAY)
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
    
    # Filter top k least blurry images
    def filter_inputs(self, k_val):
        input_blur_list = list()
        for tensor_clip in self.images:
            clip_blur_score = self.compute_clip_blur(tensor_clip)
            input_blur_list.append(clip_blur_score)
        top_k_idxs = sorted(range(len(input_blur_list)), key=lambda i: input_blur_list[i], reverse=True)[:k_val]
        filtered_inputs = self.images[top_k_idxs]
        return filtered_inputs
