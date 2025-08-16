import cv2
import numpy as np

def resize_image(image, max_dim=600):
    """
    Resizes an image if its dimensions exceed max_dim, preserving aspect ratio.
    """
    (h, w) = image.shape[:2]
    
    if h > max_dim or w > max_dim:
        # Find the scaling factor
        if h > w:
            ratio = max_dim / float(h)
            new_dim = (int(w * ratio), max_dim)
        else:
            ratio = max_dim / float(w)
            new_dim = (max_dim, int(h * ratio))
            
        # Perform the resizing using an efficient interpolation method
        resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
        print(f"Image resized from {w}x{h} to {resized.shape[1]}x{resized.shape[0]}")
        return resized
        
    return image

def get_average_color_mean(image):
    """Calculates the average color using the mean of all pixels."""
    average_color = np.mean(image, axis=(0, 1))
    return tuple(map(int, average_color))

def get_dominant_color_kmeans(image, k=3):
    """
    Finds the most dominant color using K-Means clustering.
    """
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = centers[np.argmax(counts)]
    
    return tuple(map(int, dominant_color))

def get_average_color(image_path, method='mean', k_clusters=3, resize_max_dim=600):
    """
    Calculates the average or dominant color of an image.
    Automatically resizes large images for performance.

    Args:
        image_path (str): The path to the image file.
        method (str): The method to use: 'mean' or 'kmeans'.
        k_clusters (int): The number of clusters for the k-means method.
        resize_max_dim (int): The maximum dimension (width or height) for resizing.

    Returns:
        tuple: A tuple representing the color in (Blue, Green, Red) format.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not open or find the image at '{image_path}'")
        return None, None

    # Resize the image for faster processing
    processed_image = resize_image(original_image, max_dim=resize_max_dim)
    
    if method == 'mean':
        color = get_average_color_mean(processed_image)
    elif method == 'kmeans':
        color = get_dominant_color_kmeans(processed_image, k=k_clusters)
    else:
        print(f"Error: Invalid method '{method}'. Choose 'mean' or 'kmeans'.")
        return None, None
        
    return color, processed_image


# --- --- --- Main Execution --- --- ---
if __name__ == "__main__":
    # --- --- --- TOGGLE SETTINGS HERE --- --- ---
    IMAGE_FILE = 'pexels-no-name-14543-66997.jpg'
    # Choose your method: 'mean' or 'kmeans'
    AVERAGING_METHOD = 'kmeans' 
    # Set the number of dominant colors to find (only for 'kmeans')
    K_CLUSTERS = 5
    # Set the maximum dimension for resizing to speed up processing
    RESIZE_DIMENSION = 600
    # --- --- -------------------------- --- --- ---
    
    avg_color, original_image = get_average_color(
        IMAGE_FILE, 
        method=AVERAGING_METHOD, 
        k_clusters=K_CLUSTERS,
        resize_max_dim=RESIZE_DIMENSION
    )

    if avg_color:
        print(f"\nImage: {IMAGE_FILE}")
        print(f"Method Used: {AVERAGING_METHOD}")
        print(f"The resulting color is (BGR): {avg_color}")

        # --- Visualize the results ---
        color_swatch = np.zeros((200, 200, 3), dtype=np.uint8)
        color_swatch[:] = avg_color
        
        cv2.imshow(f'Resulting Color ({AVERAGING_METHOD})', color_swatch)
        cv2.imshow('Original Image', original_image)
        
        print("\nPress any key to close the windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()