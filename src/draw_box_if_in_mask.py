from PIL import Image
from typing import Tuple

def is_center_in_white_area(image_path: str,
                            coords: Tuple[float,
                                          float,
                                          float,
                                          float]) -> int:
    """
    Checks if the center of the given rectangle is in the white area of the mask image.
    
    Args:
    image_path (str): The path to the mask image.
    coords (Tuple[float, float, float, float]): A tuple containing the coordinates of the rectangle (x1, y1, x2, y2).
    
    Returns:
    int: 1 if the center is in the white area, 0 otherwise.
    """
    image = Image.open(image_path)
    
    # Get the center of the rectangle
    x1, y1, x2, y2 = coords
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Get the pixel value at the center
    pixel_value = image.getpixel((center_x, center_y))
    
    # Check if the pixel is white (255)
    return 1 if pixel_value == 255 else 0