
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import cv2

class PestControlDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, patch_size=256, stride=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.tif', '.tiff'))]
        self.patches = []
        
        # Pre-calculate patches
        for img_file in self.image_files:
            img_path = os.path.join(images_dir, img_file)
            
            # Use rasterio to get dimensions without loading data
            with rasterio.open(img_path) as src:
                H, W = src.height, src.width
            
            # Check mask existence
            mask_name = os.path.splitext(img_file)[0]
            mask_path = None
            for ext in ['.png', '.tif', '.tiff', '.jpg']:
                potential_path = os.path.join(masks_dir, mask_name + ext)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break
            
            if mask_path is None:
                print(f"Warning: Mask for {img_file} not found. Skipping.")
                continue

            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.patches.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'x': x,
                        'y': y
                    })
        
        print(f"Found {len(self.patches)} patches from {len(self.image_files)} images.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        x, y = patch_info['x'], patch_info['y']
        
        # Load image patch
        with rasterio.open(patch_info['img_path']) as src:
             window = Window(x, y, self.patch_size, self.patch_size)
             img = src.read(window=window) # (C, H, W)
        
        # Load mask patch
        # If mask is georeferenced TIF
        if patch_info['mask_path'].endswith(('.tif', '.tiff')):
             with rasterio.open(patch_info['mask_path']) as src:
                 window = Window(x, y, self.patch_size, self.patch_size)
                 mask = src.read(1, window=window)
        else:
             # Standard image (assuming same size as source image)
             # This is tricky if mask is huge PNG and we only want a patch
             # Often better to use tiled TIFs for masks too
             # For now, load full mask and crop (inefficient for large files)
             full_mask = cv2.imread(patch_info['mask_path'], cv2.IMREAD_GRAYSCALE)
             mask = full_mask[y:y+self.patch_size, x:x+self.patch_size]
        
        # Preprocessing
        img = img.astype(np.float32)
        # normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            # Apply transforms if any
            pass

        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).long() # Use long for CrossEntropy, float for BCE
        
        return img_tensor, mask_tensor
