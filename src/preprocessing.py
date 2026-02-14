
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import os
import cv2

class MultispectralImage:
    """
    Handles loading and preprocessing of multispectral drone imagery.
    Assumes input is a 5-band TIFF or similar format:
    Band 1: Blue
    Band 2: Green
    Band 3: Red
    Band 4: Red Edge
    Band 5: Near Infrared (NIR)
    
    Adjust BAND_MAP if your channel order differs.
    """
    
    BAND_MAP = {
        'Blue': 0,
        'Green': 1,
        'Red': 2,
        'RedEdge': 3,
        'NIR': 4
    }

    def __init__(self, filepath, load_full=True):
        self.filepath = filepath
        self.data = None
        self.profile = None
        if load_full:
            self.data, self.profile = self._load_image()
        
    def _load_image(self):
        with rasterio.open(self.filepath) as src:
            data = src.read() # (Bands, Height, Width)
            profile = src.profile
        return data, profile

    def read_window(self, x, y, width, height):
        from rasterio.windows import Window
        with rasterio.open(self.filepath) as src:
            window = Window(x, y, width, height)
            data = src.read(window=window)
        return data.astype(np.float32)


    def get_band(self, band_name):
        idx = self.BAND_MAP.get(band_name)
        if idx is None:
            raise ValueError(f"Band {band_name} not found in BAND_MAP")
        return self.data[idx].astype(np.float32)

    def calculate_ndvi(self):
        """(NIR - Red) / (NIR + Red)"""
        nir = self.get_band('NIR')
        red = self.get_band('Red')
        numerator = nir - red
        denominator = nir + red + 1e-8 # Avoid division by zero
        return numerator / denominator

    def calculate_ndre(self):
        """(NIR - RedEdge) / (NIR + RedEdge)"""
        nir = self.get_band('NIR')
        re = self.get_band('RedEdge')
        numerator = nir - re
        denominator = nir + re + 1e-8
        return numerator / denominator

    def calculate_gndvi(self):
        """(NIR - Green) / (NIR + Green)"""
        nir = self.get_band('NIR')
        green = self.get_band('Green')
        numerator = nir - green
        denominator = nir + green + 1e-8
        return numerator / denominator

    def get_rgb(self):
        """Returns standard RGB image for visualization (Red, Green, Blue)"""
        r = self.get_band('Red')
        g = self.get_band('Green')
        b = self.get_band('Blue')
        
        rgb = np.dstack((r, g, b))
        
        # Normalize to 0-255 for display if needed, or 0-1
        # Simple min-max normalization for visualization
        rgb_norm = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-8)
        return (rgb_norm * 255).astype(np.uint8)

    def normalize(self):
        """Normalize all bands to 0-1 range"""
        data_norm = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data) + 1e-8)
        return data_norm

def create_patches(image, label_mask=None, patch_size=256, stride=256):
    """
    Splits image (C, H, W) and optional mask (H, W) into patches.
    """
    patches_img = []
    patches_mask = []
    
    C, H, W = image.shape
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches_img.append(patch)
            
            if label_mask is not None:
                mask_patch = label_mask[y:y+patch_size, x:x+patch_size]
                patches_mask.append(mask_patch)
                
    return np.array(patches_img), np.array(patches_mask) if label_mask is not None else None
