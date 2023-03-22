import unittest
import numpy as np
import cv2

class TestImageConversion(unittest.TestCase):
    
    def test_image_conversion(self):
        # Load the image
        image = cv2.imread("Images/bear/input.jpg")
        
        # Check if the image is not None
        self.assertIsNotNone(image, "Failed to read image")
        # Check if the image is read as numpy array format
        self.assertIsInstance(image, np.ndarray, "Image is not a NumPy array.")
        # Check if the image is in uint8 format
        self.assertEqual(image.dtype, np.uint8, "Image is not of uint8 type.")

if __name__ == "__main__":
    unittest.main()
