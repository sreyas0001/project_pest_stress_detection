
import rasterio
import cv2
import os

base_path = r"c:\Users\Sreyas SS\Documents\Anti Gravity\Maze_project\data\train\images"
files = [
    "IMG_250923_062403_0000_GRE.TIF",
    "IMG_250923_062403_0000_NIR.TIF",
    "IMG_250923_062403_0000_RED.TIF",
    "IMG_250923_062403_0000_REG.TIF",
    "IMG_250923_062403_0000_RGB.JPG"
]

print("Checking file metadata:")
for f in files:
    path = os.path.join(base_path, f)
    if not os.path.exists(path):
        print(f"File not found: {f}")
        continue
        
    try:
        if f.lower().endswith('.tif'):
            with rasterio.open(path) as src:
                print(f"{f}: Shape={src.shape}, Count={src.count}, Dtype={src.dtypes}")
        else:
            img = cv2.imread(path)
            if img is None:
                 print(f"{f}: Failed to load with cv2")
            else:
                 print(f"{f}: Shape={img.shape}, Dtype={img.dtype}")
    except Exception as e:
        print(f"{f}: Error {e}")
