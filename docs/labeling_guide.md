# Image Labeling Guide for Pest Detection

## 1. Description
Labeling is the process of marking regions in your images that contain pests or stress. This creates a "Ground Truth" dataset that the AI uses to learn.

## 2. Recommended Tools
We recommend using **LabelMe**, a simple and open-source graphical image annotation tool.

### Installation
If you have Python installed:
```bash
pip install labelme
```

## 3. Labeling Process

### Step 1: Prepare Your Images
Since your data consists of separate spectral bands and an RGB JPEG, it is easiest for humans to label the **RGB JPEG** (`_RGB.JPG`) images, as they look like normal photos.

1.  Launch LabelMe:
    ```bash
    labelme
    ```
2.  Click **"Open Dir"** and navigate to your `data/train/images` directory.
3.  In the file list (bottom right), select an `_RGB.JPG` image.

### Step 2: Annotate
1.  Click **"Create Polygons"** (or right-click the image -> Create Polygons).
2.  Click around the border of a pest-affected or stressed area.
3.  Close the polygon by clicking the first point again.
4.  A dialog will appear. Enter the label name:
    *   **"pest"** (for infested areas)
    *   **"stress"** (if distinguishing general stress)
    *   **"healthy"** (optional, usually everything else is background)
5.  Click **OK**.
6.  Repeat for all affected areas in the image.

### Step 3: Save
1.  Click **Save**. This will create a `.json` file with the same name as the image (e.g., `IMG_..._RGB.json`).
2.  Save this JSON file in the same directory (`data/train/images`) or a dedicated `data/train/labels` directory if you prefer to keep them separate.

## 4. Next Steps
Once you have labeled the images:
1.  We will write a script to convert the JSON annotations into **Binary Masks** (black and white images where white = pest).
2.  These masks will correspond to the pixel coordinates in the multispectral TIFFs (assuming the RGB and TIFFs are aligned).

## Note on Alignment
This process assumes the `_RGB.JPG` images are aligned with the spectral TIFF bands. If they are not (e.g., the RGB camera has a slight offset from the multispectral sensors), we may need to align them first or label directly on a False Color Composite generated from the TIFFs.
