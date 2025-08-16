# 🎨 Cinema Palette Generator

A Python script that analyzes video or image files to generate a visual color palette. For videos, it creates a "barcode" that represents the film's color journey from start to finish. For images, it finds the single most representative color.



---

## ✨ Features

* **Dual Media Support**: Works seamlessly with both **video files** (MP4, AVI, MOV) and **image files** (JPG, PNG, WEBP).
* **Two Analysis Methods**:
    * **K-Means (Dominant Color)**: Finds the most visually prominent color, resulting in a vibrant and representative palette. (Recommended)
    * **Mean (Average Color)**: Calculates the simple mathematical average of all pixels, which is faster but can sometimes produce "muddy" colors.
* **Performance Optimized**: Automatically resizes large images and video frames before analysis to ensure fast processing without sacrificing accuracy.
* **High-Quality Output**: Generates a high-resolution, fixed-size palette image for videos, perfect for sharing or display.
* **Highly Configurable**: Easily tweak settings like sample rate, palette dimensions, and analysis method directly in the script.

---

## 🖼️ Output Examples

### Video Palette

The script condenses an entire video into a single, beautiful image that shows the evolution of its color scheme.

**Input:** `barbie_trailer.mp4`
**Output:** `palette.png`


### Image Analysis

For a single image, it displays the original alongside a swatch of its most dominant or average color.

**Input:** `your_image.jpg`
**Output:** An OpenCV window showing the image and the color.


---

## ⚙️ How It Works

The script processes media files in a few key steps:

1.  **File Detection**: It first checks the file extension to determine if the input is an image or a video.
2.  **Frame Sampling (for Videos)**: To analyze videos efficiently, the script samples frames at a configurable rate (e.g., one frame per second) instead of processing every single one.
3.  **Color Analysis**: For each frame or image, it calculates the representative color using one of two methods:
    * **K-Means**: Groups all pixel colors into a set number of clusters (`k`) and selects the center of the largest cluster as the most dominant color.
    * **Mean**: A straightforward mathematical average of the Red, Green, and Blue values of all pixels.
4.  **Palette Generation**: The collected colors are then drawn as vertical columns onto a new, high-resolution image canvas. The width of each column is calculated to ensure the final palette perfectly fills the desired dimensions (e.g., 1920x1080).

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* Git

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone [https://github.com/tylerg/Cinema-Barcode-Generator.git](https://github.com/tylerg/Cinema-Barcode-Generator.git)
cd Cinema-Barcode-Generator
```

### 2. Install Dependencies

This script relies on `OpenCV` and `NumPy`. You can install them using pip. It's recommended to do this in a virtual environment.

```bash
pip install opencv-python numpy
```

### 3. Configure and Run

All configuration is done by editing the main execution block at the bottom of the `main.py` script.

```python
# --- --- --- CONFIGURE SETTINGS HERE --- --- ---

# Provide the path to your image OR video file
FILE_PATH = 'your_video.mp4'

# General settings
# 'kmeans' is recommended for more vibrant colors. Use 'mean' for a faster, simpler average.
AVERAGING_METHOD = 'kmeans'
RESIZE_DIMENSION = 400       # Smaller values are faster

# K-Means specific settings
K_CLUSTERS = 5

# Video specific settings
# Analyzes 1 frame every X frames. For 30fps video, 30 = ~1 sample per second.
VIDEO_SAMPLE_RATE = 30

# Final palette output dimensions
PALETTE_WIDTH = 1920
PALETTE_HEIGHT = 1080

# --- ------------------------------------- --- ---
```

Once you've configured the settings, simply run the script:

```bash
python main.py
```

An OpenCV window will pop up to display the result. Press any key to close it. The script does not save the output file automatically, but you can easily add a `cv2.imwrite("palette.png", palette)` line to the `process_video` function to save the result.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/tylerg/Cinema-Barcode-Generator/issues).

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
