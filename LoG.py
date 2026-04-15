"""LOG Blob detection Example"""

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import blob_log
import argparse, os, time, random

def parse_option():
    parser = argparse.ArgumentParser('LoG', add_help=False)
    parser.add_argument('floder_path', default="./", type=str, help='Path to the folder containing input images')
    parser.add_argument('save_path', default="./", type=str, help='Path to the folder where results will be saved')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_option()
    folder_path = args.floder_path
    save_path = args.save_path
    files_root_W = [f for f in os.listdir(folder_path)]

    for i in tqdm(range(len(files_root_W))):
        """Read Image"""
        root_W = os.path.join(folder_path, files_root_W[i])
        image_W = Image.open(root_W).convert("RGB")
        image_W_np = np.array(image_W)
        gray_image_W = image_W.convert("L")

        """Blob Detection using LOG (hyper parameters may need to be adjusted)"""
        blobs = blob_log(
            gray_image_W, min_sigma=8, max_sigma=10, num_sigma=10, threshold=0.1
        )
        blobs[:, 2] = 10

        """Filter blobs if less than 100"""
        if len(blobs) < 100:
            print(f"Found only {len(blobs)} blobs, skipping this image.")
            continue

        """Create image with only detected blobs"""
        image = Image.open(root_W).convert("RGB")
        image_np = np.array(image)
        gray_image = image.convert("L")
        gray_np = np.array(gray_image)
        rows, cols = gray_np.shape



        Y, X = np.ogrid[:rows, :cols]
        image_white = np.zeros_like(image_np) * 255
        image_intensities = np.zeros_like(gray_np) 
        avg_intensities = [] 
        for y, x, r in tqdm(blobs, desc="Processing blobs"):
            mask = (X - x) ** 2 + (Y - y) ** 2 <= (r * 0.8) ** 2                    # Locate the position of the atom
            image_np[mask] = [255, 0, 0]                                            # Set the masked area to red
            image_white[mask] = [255, 255, 255]                                     # Set the masked area to white
            mean_val = gray_np[mask].mean() 
            image_intensities[mask] = mean_val                                      # Atomic internsities
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
        path = os.path.join(save_path_single, "atoms_with_circles.png")             # Mark the atomic positions on the original image with red.
        plt.savefig(path, bbox_inches='tight', pad_inches=0) 



        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image_white) 
        plt.axis("off")
        plt.tight_layout(pad=0)
        path = os.path.join(save_path_single, "only_atoms.png")                     # Mark the atomic positions.
        plt.savefig(path, bbox_inches='tight', pad_inches=0) 


        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image_intensities, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        path = os.path.join(save_path_single, "atoms_intensities.png")              # Mark the atomic positions and internsities.
        plt.savefig(path, bbox_inches='tight', pad_inches=0) 


        path = os.path.join(save_path_single, "denoise.csv")
        np.savetxt(path, image_intensities, delimiter=',', fmt='%.0f')              # save the atomic positions and internsities (csv).
        plt.close('all')
        
        avg_intensities = np.array(avg_intensities).reshape(-1, 1)
        blobs_with_intensity = np.hstack([blobs, avg_intensities]) 
        path = os.path.join(save_path_single, "blobs_with_intensity.csv")           # only save the atomic positions and internsities (csv).
        np.savetxt(path, blobs_with_intensity, delimiter=',', header='y,x,r,intensity', comments='', fmt='%.0f')
