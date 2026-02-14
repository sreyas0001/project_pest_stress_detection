
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
        'Green': 0,
        'Red': 1,
        'RedEdge': 2,
        'NIR': 3
    }

    def __init__(self, filepath, load_full=True):
        """
        filepath: Path to ANY file in the set (e.g., ..._NIR.TIF or ..._RGB.JPG).
                  The class will automatically find the sibling band files based on the prefix.
        """
        self.base_path = self._get_base_path(filepath)
        self.data = None
        self.profile = None
        self.rgb_path = self.base_path + "_RGB.JPG"
        
        if load_full:
            self.data, self.profile = self._load_image()
        
    def _get_base_path(self, filepath):
        # Assumes format: path/to/IMG_date_time_idx_BAND.EXT
        # We want: path/to/IMG_date_time_idx
        # Split by underscore and reassemble until the band identifier
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        
        # Heuristic: split by '_' and assume the last part is the band/ext
        # Example: IMG_250923_062403_0000_GRE.TIF -> prefix is everything before _GRE
        parts = basename.rsplit('_', 1)
        if len(parts) < 2:
             raise ValueError(f"Filename {basename} does not match expected format IMG_..._BAND.ext")
        
        prefix = parts[0]
        return os.path.join(dirname, prefix)

    def _load_image(self):
        # Load bands in order of BAND_MAP
        bands = []
        profile = None
        
        # Map internal indices to suffixes
        # 0: Green -> _GRE.TIF
        # 1: Red   -> _RED.TIF
        # 2: RE    -> _REG.TIF
        # 3: NIR   -> _NIR.TIF
        suffix_map = {
            0: '_GRE.TIF',
            1: '_RED.TIF',
            2: '_REG.TIF',
            3: '_NIR.TIF'
        }

        first_band_loaded = False
        
        for idx in range(len(self.BAND_MAP)):
            suffix = suffix_map[idx]
            band_path = self.base_path + suffix
            
            if not os.path.exists(band_path):
                # Try .tif lowercase just in case
                band_path = self.base_path + suffix.replace('.TIF', '.tif')
                if not os.path.exists(band_path):
                    raise FileNotFoundError(f"Band file not found: {band_path}")

            with rasterio.open(band_path) as src:
                band_data = src.read(1) # Read the first (and only) band
                bands.append(band_data)
                
                if not first_band_loaded:
                    profile = src.profile
                    first_band_loaded = True
        
        # Stack into (Count, H, W)
        data = np.stack(bands, axis=0)
        
        # Update profile to reflect new band count
        profile.update(count=data.shape[0])
        
        return data, profile

    def read_window(self, x, y, width, height):
        from rasterio.windows import Window
        window = Window(x, y, width, height)
        
        bands = []
        suffix_map = {0: '_GRE.TIF', 1: '_RED.TIF', 2: '_REG.TIF', 3: '_NIR.TIF'}
        
        for idx in range(len(self.BAND_MAP)):
            suffix = suffix_map[idx]
            band_path = self.base_path + suffix
            if not os.path.exists(band_path):
                 band_path = self.base_path + suffix.replace('.TIF', '.tif')
            
            with rasterio.open(band_path) as src:
                b = src.read(1, window=window)
                bands.append(b)
                
        return np.stack(bands).astype(np.float32)

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
        """
        Returns RGB image.
        Since we don't have a Blue TIF band, we try to load the _RGB.JPG.
        If unavailable, we return a False Color Composite (NIR, Red, Green).
        """
        if os.path.exists(self.rgb_path):
            # Load the reference RGB JPG
            # Note: This might have different resolution/registration than TIFs
            img = cv2.imread(self.rgb_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Fallback to False Color (NIR, Red, Green) which is commonly used in Ag
        nir = self.get_band('NIR')
        red = self.get_band('Red')
        green = self.get_band('Green')
        
        fcc = np.dstack((nir, red, green))
        fcc_norm = (fcc - np.min(fcc)) / (np.max(fcc) - np.min(fcc) + 1e-8)
        return (fcc_norm * 255).astype(np.uint8)

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
