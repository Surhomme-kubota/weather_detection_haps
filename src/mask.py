
import cv2
import numpy as np


class CreateFisheyeMaskWithAnnotations:
    def __init__(self, image_path: str, azimuth_min_deg: float, azimuth_max_deg: float, elevation_min_deg: float, elevation_max_deg: float, output_path: str):
        """
        CreateFisheyeMaskWithAnnotations class constructor.

        Args:
            image_path (str): Path to the input fisheye image.
            azimuth_min_deg (float): Minimum azimuth angle in degrees.
            azimuth_max_deg (float): Maximum azimuth angle in degrees.
            elevation_min_deg (float): Minimum elevation angle in degrees.
            elevation_max_deg (float): Maximum elevation angle in degrees.
            output_path (str): Path to save the annotated mask image.
        """
        self.image_path = image_path
        self.azimuth_min_deg = azimuth_min_deg
        self.azimuth_max_deg = azimuth_max_deg
        self.elevation_min_deg = elevation_min_deg
        self.elevation_max_deg = elevation_max_deg
        self.output_path = output_path

    def process(self, data: any = None) -> None:
        """
        Create a fisheye mask with annotations for the specified azimuth and elevation ranges.

        Args:
            data (any): Placeholder for the initial data passed through the pipeline.

        Returns:
            None
        """
        # Load the fisheye image
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Convert degrees to radians
        azimuth_min = np.deg2rad(self.azimuth_min_deg)
        azimuth_max = np.deg2rad(self.azimuth_max_deg)
        elevation_min = np.deg2rad(self.elevation_min_deg)
        elevation_max = np.deg2rad(self.elevation_max_deg)

        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate the radius (assuming fisheye projection)
        radius = min(center_x, center_y)

        # Create the mask based on the given azimuth and elevation ranges
        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = center_y - y  # North is at the top

                distance = np.sqrt(dx**2 + dy**2)
                if distance == 0 or distance > radius:
                    continue

                theta = np.arctan2(dx, dy)  # Azimuth angle from north
                if theta < 0:
                    theta += 2 * np.pi  # Normalize theta to be in the range [0, 2*pi]

                phi = np.arccos(distance / radius)  # Elevation angle

                if azimuth_min <= theta <= azimuth_max and elevation_min <= phi <= elevation_max:
                    mask[y, x] = 255

        # Draw a marker at the north origin (top center of the image)
        north_origin_y = 0  # North is at the top of the image
        north_origin_x = center_x
        cv2.circle(mask, (north_origin_x, north_origin_y), 10, (127), -1)  # Draw in gray for visibility

        # Annotate azimuth and elevation ranges
        for theta in np.linspace(azimuth_min, azimuth_max, 100):
            for phi in np.linspace(elevation_min, elevation_max, 100):
                x = int(center_x + radius * np.sin(phi) * np.sin(theta))
                y = int(center_y - radius * np.sin(phi) * np.cos(theta))
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 127

        # Save the annotated mask image
        cv2.imwrite(self.output_path, mask)
        
        # Mask Area
        mask_area = np.sum(mask == 255)
        print(mask_area)
