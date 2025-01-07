import cv2
import numpy as np

class ApplyMaskToImage:
    def __init__(self,
                 image_path: str,
                 mask_path: str,
                 output_path: str) -> None: 
        """
        ApplyMaskToImage class constructor.

        Parameters:
            image_path (str): Path to the image to be processed.
            mask_path (str): Path to the mask image.
            output_path (str): Path to save the masked image.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path

    def process(self, data: any) -> int:
        """
        Applies a mask to an image and saves the result. Counts the white pixels in the mask.

        Parameters:
            data (any): Placeholder for the initial data passed through the pipeline.

        Returns:
            int: The number of white pixels in the mask.

        Raises:
            FileNotFoundError: If the image or mask file cannot be found.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"The image file at {self.image_path} could not be found.")
        
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"The mask file at {self.mask_path} could not be found.")

        # Apply threshold to create a binary mask
        _, thresholded_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=thresholded_mask)
        
        # Save the resulting masked image
        cv2.imwrite(self.output_path, masked_image)
        print(f"Masked image saved to {self.output_path}")

        # Count the number of white pixels in the mask
        num_white_pixels = np.sum(thresholded_mask == 255)
        return num_white_pixels