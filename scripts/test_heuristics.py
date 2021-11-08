import torch
from heuristics import Blur

if __name__ == '__main__':
    random_tensor = torch.rand((44,4,3,224,224))
    blur_filter = Blur(random_tensor)
    top_k_idxs = blur_filter.get_least_blurry(4)
    print(top_k_idxs)

