from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from sfin import SFIN
import numpy as np
from scipy import ndimage
from utils import *
import numpy as np
import argparse, os, time, random
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('AtomSegementation Training', add_help=False)
    parser.add_argument('floder_path', default="./", type=str, help='Path to the folder containing input images')
    parser.add_argument('save_path', default="./", type=str, help='Path to the folder where results will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/SFIN_RealSTEM255.pth', help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_option()
    folder_path = args.floder_path
    save_path = args.save_path
    CUDA_VISIBLE_DEVICES=args.gpu

    """Set paths and prepare file lists (files_root_W are filted images with more details)"""
    files_root_W = [f for f in os.listdir(folder_path)]
    unet = SFIN().cuda(device=0)
    weights = torch.load(args.checkpoint_dir)
    unet.load_state_dict(weights)
    unet.eval()  



    for i in tqdm(range(len(files_root_W))):
        """Read Image"""
        in_path = os.path.join(folder_path, files_root_W[i])
        out_path = os.path.join(save_path, f"{files_root_W[i]}.png")
        if "tif" not in in_path:
            in_img = torchvision.io.read_image(in_path).cuda(0)
            if in_img.shape[0] == 3:
                in_img = in_img[:1]
            elif in_img.shape[0] == 4:
                print("RGBA image found, converting to grayscale.")
                continue
        else:
            import tifffile as tiff
            img = tiff.imread(in_path)
            in_img = torch.from_numpy(img).float().cuda(0)
            if len(in_img.shape) == 3:
                in_img = in_img[:, :, 0].unsqueeze(0)
            else:
                in_img = in_img.unsqueeze(0)
        _, H, W = in_img.shape


        """Forward propagation"""
        patch = torch.unsqueeze(in_img, 0).float()
        pred_patch = unet(patch)
        pred_patch = torch.clip_(pred_patch, 0, 255)
        pred_patch = torch.squeeze(pred_patch, 0).byte()
        pred_patch[pred_patch > 128] = 255
        pred_patch[pred_patch < 128] = 0

        mask = pred_patch // 255
        _, rows, cols = mask.shape
        torchvision.io.write_png(pred_patch.cpu(), out_path)


        """Delete the edge points"""
        # if rows == 2048:
        #     y_min, y_max = 96, 1952
        #     x_min, x_max = 96, 1952
        # else:
        #     y_min, y_max = 192, 3904
        #     x_min, x_max = 192, 3904

        

        """Identify the connected area"""
        labeled_array, num_features = ndimage.label(mask.squeeze().detach().cpu().numpy())
        centers = ndimage.center_of_mass(mask.squeeze().detach().cpu().numpy(), labeled_array, range(1, num_features + 1))
        areas = ndimage.sum(mask.squeeze().detach().cpu().numpy(), labeled_array, range(1, num_features+1)) 
        areas = areas[areas > 150]
        r = np.mean(np.sqrt(areas / np.pi)).astype(int)



        image = Image.open(in_path).convert("RGB") 
        image_np = np.array(image)
        gray_image = image.convert("L")
        gray_np = np.array(gray_image)
        avg_intensities, ra = [], [] 
        Y, X = np.ogrid[:rows, :cols]
        image_white = np.zeros_like(image_np) * 255
        image_intensities = np.zeros_like(gray_np) 
        for y, x in tqdm(centers, desc="Processing blobs"):
            ra.append(r)
            """Delete the edge points"""
            # if y < y_min or y > y_max or x < x_min or x > x_max:
            #     avg_intensities.append(0) 
            #     continue
            mask_ = (X - x) ** 2 + (Y - y) ** 2 <= (r) ** 2                 # Locate the position of the atom
            image_np[mask_] = [255, 0, 0]                                   # Set the masked area to red
            image_white[mask_] = [255, 255, 255]                            # Set the masked area to white
            mean_val = gray_np[mask_].mean() 
            image_intensities[mask_] = mean_val                             # Atomic internsities
            avg_intensities.append(mean_val) 
            

        """save"""
        save_path_single = os.path.join(save_path, files_root_W[i][:-4])
        os.makedirs(save_path_single, exist_ok=True)
        os.makedirs(os.path.join(save_path_single, 'normal'), exist_ok=True)

        height, width = image_np.shape[:2] 
        dpi = 100 
        figsize = (width / dpi, height / dpi) 
        plt.figure(figsize=figsize, dpi=dpi) 
        plt.imshow(image_np) 
        plt.axis('off') 
        plt.tight_layout(pad=0)
        path = os.path.join(save_path_single, "atoms_with_circles.png")     # Mark the atomic positions on the original image with red.
        plt.savefig(path, bbox_inches='tight', pad_inches=0) 


        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image_white)
        plt.axis('off')
        plt.tight_layout(pad=0)
        path = os.path.join(save_path_single, "only_atoms.png")             # Mark the atomic positions.
        plt.savefig(path, bbox_inches='tight', pad_inches=0) 


        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image_intensities, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        path = os.path.join(save_path_single, "atoms_intensities.png")      # Mark the atomic positions and internsities.
        plt.savefig(path, bbox_inches='tight', pad_inches=0) 


        path = os.path.join(save_path_single, "denoise.csv")
        np.savetxt(path, image_intensities, delimiter=',', fmt='%.0f')      # save the atomic positions and internsities (csv).
        plt.close('all')
        avg_intensities = np.array(avg_intensities).reshape(-1, 1)
        ra = np.array(ra).reshape(-1, 1)
        blobs_with_intensity = np.hstack([centers, ra, avg_intensities]) 
        path = os.path.join(save_path_single, "blobs_with_intensity.csv")   # only save the atomic positions and internsities (csv).
        np.savetxt(path, blobs_with_intensity, delimiter=',', header='y,x,r,intensity', comments='', fmt='%.0f')

        