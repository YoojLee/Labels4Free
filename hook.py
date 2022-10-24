# Calculate which layer is most responsible for generating the background pixels in a StyleGAN image
import os, glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from model_new import Generator
from dataset import TestDataset



class Hook(object):
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # forward / backward pass 시 각 layer 값을 확인
        self.input = input
        self.output = output
    
    def close(self):
        self.hook.remove()

def get_gradient(net, bg_crops, mean_latent, device):

    hookB = {layer_name: Hook(layer, backward=True) for layer_name, layer in net.named_modules()}

    
    sample_z = torch.randn(bg_crops.shape[0], 512, device=device)

    # forward
    out, _ = net(
        [sample_z], truncation=0.5, truncation_latent=mean_latent
    )

    # compute loss
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    loss = loss_fn(bg_crops, out)

    loss.backward(torch.ones_like(loss, dtype=torch.float), retain_graph=True)

    return hookB

class ResizedCrop(object):
    def __init__(self, resize, x,y,h,w):
        self.resize = resize
        self.x = x
        self.y = y
        self.h = h
        self.w = w
    
    def __call__(self, img):
        cropped = F.crop(img, self.x, self.y, self.h, self.w)
        transform = transforms.Compose(
            [
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
            ]
        )

        return transform(cropped)

if __name__ == "__main__":
    
    device = "cuda:0"
    path = "sample/"
    
    g_ema = Generator(256, 512, 8, channel_multiplier=1).to(device)
    checkpoint = torch.load("/home/workspace/checkpoint/flowers-256-slim-001212.pt")

    g_ema.load_state_dict(checkpoint["g_ema"])

    with torch.no_grad():
        mean_latent = g_ema.mean_latent(4096)

    transform = ResizedCrop(256, 0, 0, 20, 20)

    dataset = TestDataset(path, transform)
    dataloader = DataLoader(dataset, batch_size=len(os.listdir(path)), pin_memory=True)

    for idx, img in enumerate(dataloader):
        img = img.to(device)

        hook_dct = get_gradient(g_ema, img, mean_latent, device)
    
    grad_sum = dict()
    for h in hook_dct.keys():
        try:
            print(f"{h}: {len(hook_dct[h].output)}")
            grad_sum[h] = hook_dct[h].output[0].sum(dim=1)
        except:
            print(h)

    
    print(sorted(grad_sum.items(), key=lambda x: x[1])[:5])