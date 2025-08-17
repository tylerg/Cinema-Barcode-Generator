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

def create_static_palette(colors, width=1920, height=1080):
    """Creates a static, barcode-style image from a list of colors."""
    num_colors = len(colors)
    if num_colors == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    col_width = width / num_colors
    
    for i, color in enumerate(colors):
        x_start = int(i * col_width)
        x_end = int((i + 1) * col_width)
        palette[:, x_start:x_end] = color
        
    # Ensure the last color extends to the very edge
    palette[:, int((num_colors - 1) * col_width):] = colors[-1]
    return palette

def create_video_animation(colors, settings):
    """Creates an animated video of the color palette appearing over time."""
    width = settings['palette_width']
    height = settings['palette_height']
    duration = settings['animation_duration']
    filename = settings['output_filename']
    fps = 30

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Use 'mp4v' for .mp4 files
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for '{filename}'")
        return

    total_animation_frames = duration * fps
    num_colors = len(colors)
    
    print(f"\nGenerating animation ({total_animation_frames} frames)...")

    for i in range(total_animation_frames):
        # Calculate how many color bars are visible at this frame
        current_colors_count = int((i + 1) / total_animation_frames * num_colors)
        
        # Get the slice of colors to display
        current_colors = colors[:current_colors_count]
        
        # Generate the palette for the current frame
        frame = create_static_palette(current_colors, width, height)
        
        # Write the frame to the video
        video_writer.write(frame)

        # Print progress
        print(f"  -> Wrote frame {i + 1}/{total_animation_frames}", end='\r')

    video_writer.release()
    print(f"\n\nAnimation saved successfully to '{filename}'")

def process_image(file_path, settings):
    """Analyzes a single image file and saves a color swatch."""
    # This function remains largely the same but now saves the output
    pass # For brevity, this is the same as the previous version

def process_video(file_path, settings):
    """Analyzes a video file to create a static or animated color palette."""
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
        if not ret: break
        
        if frame_count % settings['video_sample_rate'] == 0:
            print(f"Analyzing frame {frame_count}/{total_frames}...", end='\r')
            processed_frame = resize_image(frame, max_dim=settings['resize_dim'])
            
            if settings['method'] == 'mean':
                color = get_average_color_mean(processed_frame)
            else:
                color = get_dominant_color_kmeans(processed_frame, k=settings['k_clusters'])
            frame_colors.append(color)
        
        frame_count += 1
    
    print("\nVideo analysis complete.                                 ")
    cap.release()

    if not frame_colors:
        print("No frames were analyzed.")
        return

    # --- Output Generation ---
    if settings['output_type'] == 'image':
        palette = create_static_palette(frame_colors, settings['palette_width'], settings['palette_height'])
        cv2.imwrite(settings['output_filename'], palette)
        print(f"Static palette saved successfully to '{settings['output_filename']}'")
        cv2.imshow('Video Color Palette', palette)
        cv2.waitKey(0)
    elif settings['output_type'] == 'video':
        create_video_animation(frame_colors, settings)
    else:
        print(f"Error: Unknown output type '{settings['output_type']}'")


# --- --- --- Main Execution --- --- ---
if __name__ == "__main__":
    # --- --- --- CONFIGURE SETTINGS HERE --- --- ---
    
    FILE_PATH = 'ssvid.net---Barbie-Main-Trailer_1080p.mp4'
    
    # NEW: Choose 'image' or 'video'
    OUTPUT_TYPE = 'video'
    OUTPUT_FILENAME = 'output_palette.mp4' # Use .png for images, .mp4 for videos
    
    # General settings
    AVERAGING_METHOD = 'mean'
    RESIZE_DIMENSION = 400
    
    # K-Means specific settings
    K_CLUSTERS = 5
    
    # Video specific settings
    VIDEO_SAMPLE_RATE = 30
    
    # Palette & Animation settings
    PALETTE_WIDTH = 1280
    PALETTE_HEIGHT = 720
    ANIMATION_DURATION = 10 # In seconds (only for video output)
    
    # --- ------------------------------------- --- ---
    
    settings = {
        'output_type': OUTPUT_TYPE,
        'output_filename': OUTPUT_FILENAME,
        'method': AVERAGING_METHOD,
        'resize_dim': RESIZE_DIMENSION,
        'k_clusters': K_CLUSTERS,
        'video_sample_rate': VIDEO_SAMPLE_RATE,
        'palette_width': PALETTE_WIDTH,
        'palette_height': PALETTE_HEIGHT,
        'animation_duration': ANIMATION_DURATION
    }
    
    # For now, this example focuses on video processing
    process_video(FILE_PATH, settings)
    
    cv2.destroyAllWindows()