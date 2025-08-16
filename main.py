import cv2
import numpy as np
import os

def resize_image(image, max_dim=600):
    """Resizes an image if its dimensions exceed max_dim, preserving aspect ratio."""
    (h, w) = image.shape[:2]
    if h > max_dim or w > max_dim:
        if h > w:
            ratio = max_dim / float(h)
            new_dim = (int(w * ratio), max_dim)
        else:
            ratio = max_dim / float(w)
            new_dim = (max_dim, int(h * ratio))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return image

def get_average_color_mean(image):
    """Calculates the average color using the mean of all pixels."""
    average_color = np.mean(image, axis=(0, 1))
    return tuple(map(int, average_color))

def get_dominant_color_kmeans(image, k=3):
    """Finds the most dominant color using K-Means clustering."""
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = centers[np.argmax(counts)]
    return tuple(map(int, dominant_color))

def create_video_palette(colors, width=1920, height=1080):
    """
    Creates a barcode-style image from a list of colors, scaled to a fixed size.
    """
    num_colors = len(colors)
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate the width of each color column
    col_width = width // num_colors
    
    for i, color in enumerate(colors):
        # Calculate start and end x-coordinates for the column
        x_start = i * col_width
        # For the last color, extend it to the edge to fill the palette
        x_end = (i + 1) * col_width if i < num_colors - 1 else width
        
        # Fill the column with the color
        palette[:, x_start:x_end] = color
        
    return palette

def process_image(file_path, settings):
    """Analyzes a single image file."""
    print(f"Processing image: {file_path}")
    original_image = cv2.imread(file_path)
    if original_image is None:
        print("Error: Could not read image.")
        return

    processed_image = resize_image(original_image, max_dim=settings['resize_dim'])
    
    if settings['method'] == 'mean':
        color = get_average_color_mean(processed_image)
    else:
        color = get_dominant_color_kmeans(processed_image, k=settings['k_clusters'])

    print(f"Resulting color (BGR): {color}")

    # --- Visualization ---
    color_swatch = np.zeros((200, 200, 3), dtype=np.uint8)
    color_swatch[:] = color
    cv2.imshow('Original Image', original_image)
    cv2.imshow(f'Resulting Color ({settings["method"]})', color_swatch)
    cv2.waitKey(0)

def process_video(file_path, settings):
    """Analyzes a video file frame by frame to create a color palette."""
    print(f"Processing video: {file_path}")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_colors = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % settings['video_sample_rate'] == 0:
            print(f"Analyzing frame {frame_count}/{total_frames}...")
            
            processed_frame = resize_image(frame, max_dim=settings['resize_dim'])
            
            if settings['method'] == 'mean':
                color = get_average_color_mean(processed_frame)
            else:
                color = get_dominant_color_kmeans(processed_frame, k=settings['k_clusters'])
            
            frame_colors.append(color)
        
        frame_count += 1
    
    cap.release()

    if not frame_colors:
        print("No frames were analyzed. Check video_sample_rate or video file.")
        return

    print("\nVideo analysis complete. Generating color palette...")
    palette = create_video_palette(
        frame_colors, 
        width=settings['palette_width'], 
        height=settings['palette_height']
    )
    cv2.imshow('Video Color Palette', palette)
    cv2.waitKey(0)


# --- --- --- Main Execution --- --- ---
if __name__ == "__main__":
    # --- --- --- CONFIGURE SETTINGS HERE --- --- ---
    
    # Provide the path to your image OR video file
    FILE_PATH = 'ssvid.net---Barbie-Main-Trailer_1080p.mp4'
    
    # General settings
    AVERAGING_METHOD = 'mean'
    RESIZE_DIMENSION = 400
    
    # K-Means specific settings
    K_CLUSTERS = 5
    
    # Video specific settings
    VIDEO_SAMPLE_RATE = 30
    
    # NEW: Final palette output dimensions
    PALETTE_WIDTH = 1920
    PALETTE_HEIGHT = 1080
    
    # --- ------------------------------------- --- ---
    
    settings = {
        'method': AVERAGING_METHOD,
        'resize_dim': RESIZE_DIMENSION,
        'k_clusters': K_CLUSTERS,
        'video_sample_rate': VIDEO_SAMPLE_RATE,
        'palette_width': PALETTE_WIDTH,
        'palette_height': PALETTE_HEIGHT
    }
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    
    file_ext = os.path.splitext(FILE_PATH)[1].lower()
    
    if file_ext in image_extensions:
        process_image(FILE_PATH, settings)
    elif file_ext in video_extensions:
        process_video(FILE_PATH, settings)
    else:
        print(f"Error: Unsupported file type '{file_ext}'.")

    cv2.destroyAllWindows()