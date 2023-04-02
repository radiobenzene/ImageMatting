import unittest
import matplotlib.pyplot as plt
import functions as F
from tkinter import filedialog
from PIL import Image, ImageOps
import os
from functions import initializeVariables
import numpy as np


class testBeyasianMatting():
    def __init__(self, img_name):
        self.img_name = img_name

        self.IMG_PATH = os.path.join("Images","input_training_lowres","{}.png".format(self.img_name))
        self.TRIMAP_PATH = os.path.join("Images","trimap_training_lowres", "Trimap2", "{}.png".format(self.img_name))
        self.TRIMAP_FILE = Image.open(self.TRIMAP_PATH)

        self.image = np.array(Image.open(self.IMG_PATH))
        self.image_trimap = np.array(ImageOps.grayscale(self.TRIMAP_FILE))

        self.img_obj = initializeVariables()

        self.testNum = 0
        self.testOK = 0
        self.testFailed = 0

    def testImgPath(self):
        print("\n==============================================\n")
        print("Test Case 1:")
        print("Start testing the image path and trimap path")

        print("----------------------------------------------")
        print("Checking the images if exist:")
        self.testNum = self.testNum + 1
        if os.path.exists(self.IMG_PATH):
            print("The path of input image:", "{}".format(self.IMG_PATH),"exists. OK.")
            self.testOK = self.testOK + 1
            try:
                print("Test if the image can be loaded")
                self.testNum = self.testNum + 1
                # attempt to open the file as an image
                img = Image.open(self.IMG_PATH)
                # if successful, print a message indicating that the file is an image
                print(f"{self.IMG_PATH} is an image. OK.")
                self.testOK = self.testOK + 1
            except IOError:
                # if the file cannot be opened as an image, print an error message
                print(f"{self.IMG_PATH} is not an image")
                # report a error if there are some exception
                raise Exception(f"{self.IMG_PATH} is not an image")
        else:
            raise Exception("Loading Failed: Please check the image path")
        
        print("----------------------------------------------")
        print("Checking the trimap if exist:")
        self.testNum = self.testNum + 1
        if os.path.exists(self.TRIMAP_PATH):
            print("The path of Trimap:", "{}".format(self.TRIMAP_PATH),"exists")
            print("Test if the image can be loaded")
            self.testOK = self.testOK + 1
            try:
                self.testNum = self.testNum + 1
                # attempt to open the file as an image
                img = Image.open(self.TRIMAP_PATH)
                # if successful, print a message indicating that the file is an image
                print(f"{self.TRIMAP_PATH} is an image")
                self.testOK = self.testOK + 1
            except IOError:
                # if the file cannot be opened as an image, print an error message
                print(f"{self.TRIMAP_PATH} is not an image")
                # report a error if there are some exception
                raise Exception(f"{self.TRIMAP_PATH} is not an image")
        else:
            raise Exception("Loading Failed: Please check the trimap path")
        print("==============================================\n")

    def testImageInfo(self):
        # print("==============================================")
        print("Test Case 2:")
        print("Start testing the image information")

        print("----------------------------------------------")
        image = np.array(Image.open(self.IMG_PATH))
        self.testNum = self.testNum + 1
        if image.shape[-1] == 3:
            print("src_Image is a 3 channels image: OK")
            self.testOK = self.testOK + 1
        else:
            raise ValueError("Failed: The source image is not a 3 channel image")
        
        image_trimap = (ImageOps.grayscale(self.TRIMAP_FILE))
        self.testNum = self.testNum + 1
        if image_trimap.mode == "L":
            print(f"{self.TRIMAP_PATH} is a grayscale image:", "OK.")
            self.testOK = self.testOK + 1
        else:
            print(f"{self.TRIMAP_PATH} is not a grayscale image:", "Fail")

        self.testNum = self.testNum + 1
        if image.shape[:2][::-1] == image_trimap.size:
            self.testOK = self.testOK + 1
            print("The image is the same size as the trimap, OK")
        else:
            raise ValueError("Failed: The image and trimap are not the same size ")
        print("==============================================\n")

    def testConvertImage(self):
        print("Test Case 3:")
        print("Start testing the convertImage")
        print("This function expects the image to be converted to a floating point number and normalised.")

        print("----------------------------------------------")
        test_image = np.random.randint(0, 256, size=(100, 100, 3), dtype='uint8')
        converted_image = F.convertImage(test_image)

        self.testNum = self.testNum + 1
        if test_image.shape == converted_image.shape:
            self.testOK = self.testOK + 1
            print("The ConvertImage Function work successfully for 3 channel image, OK")

            self.testNum = self.testNum + 1
            if np.all((converted_image >= 0) & (converted_image <= 1)):
                self.testOK = self.testOK + 1
                print("The image has been normalised,OK.")
            else:
                raise ValueError("Failed: Normalisation failed for 3 channel image")

        else:
            raise ValueError("Failed: The convertImage Function failed, the reason is the shape of input and output is different.")

        
        test_trimap = np.random.randint(0, 256, size=(100, 100), dtype='uint8')
        converted_trimap = F.convertImage(test_trimap)
        self.testNum = self.testNum + 1
        if test_trimap.shape == converted_trimap.shape:
            self.testOK = self.testOK + 1
            print("The ConvertImage Function work successfully for trimap, OK")

            self.testNum = self.testNum + 1
            if np.all((converted_trimap >= 0) & (converted_trimap <= 1)):
                self.testOK = self.testOK + 1
                print("The trimap has been normalised,OK.")
            else:
                raise ValueError("Failed: Normalisation failed for trimap")

        else:
            raise ValueError("Failed: The convertImage Function failed, the reason is the shape of input and output is different.")
        print("==============================================\n")

    def test_fspecial(self):
        print("Test Case 4:")
        print("Start testing the fspecial function")
        print("This test is to test the Gaussian Kernal")

        print("----------------------------------------------")

        gaussian_weights = F.fspecial((3,3),1)
        except_shape = (3,3)
        self.testNum = self.testNum + 1
        if gaussian_weights.shape == except_shape:
            self.testOK = self.testOK + 1
            print("The shape of the Gaussian kernel meets expectations. OK")
        else:
            raise ValueError("Failed: The shape of the Gaussian kernel doesn't meet expectations")
        
        result = F.fspecial((3, 3), 1)
        self.testNum += 1
        if (np.sum(result) - 1.0 < 1e-6) or (1.0 - np.sum(result) < 1e-6):
            self.testOK += 1
            print("The gaussian kernel is normalized correctly. OK")
        else:
            raise ValueError("Failed: The Gaussian kernel doesn't be normalized correctly")
        
        print("==============================================\n")

    def test_displayImage(self):
        print("Test Case 5:")
        print("Start testing the fspecial function")
        print("This test is to test the Gaussian Kernal")

        print("----------------------------------------------")

        title = "Test Title"
        self.testNum += 1
        result_title, _ = displayImage(title, self.image)
        assert result_title == title, f"Expected title '{title}', but got '{result_title}'"
        self.testOK += 1
        print("displayImage works, OK")

        title = "Test Title"
        self.testNum += 1
        _, result_img = displayImage(title, self.image_trimap)
        assert np.array_equal(result_img, self.image_trimap), "Displayed image data does not match the input image data"
        self.testOK += 1
        print("displayImage return correct image")

        print("==============================================\n")

    def test_node_creation(self):
        print("Test Case 6:")
        print("Start testing the fspecial function")
        print("This test is to test the Gaussian Kernal")

        print("----------------------------------------------")
        matrix = np.random.rand(4, 3)
        w = np.array([0.25, 0.25, 0.25, 0.25])
        node = F.Node(matrix, w)
        assert node.X.shape == matrix.shape, f"Expected shape {matrix.shape}, but got {node.X.shape}"
        assert node.w.shape == w.shape, f"Expected shape {w.shape}, but got {node.w.shape}"
        print("T6: OK")
        print("==============================================\n")

    def test_cluster_elements_no_split(self):
        print("Test Case 6:")
        print("Start testing the fspecial function")
        print("This test is to test the Gaussian Kernal")

        print("----------------------------------------------")
        S = np.random.rand(50, 3)
        w = np.ones(50) / 50.0
        min_var = 10.  # Large value to prevent splitting
        mu, sigma = F.clusterElements(S, w, min_var)
        self.testNum += 1
        assert mu.shape == (1, S.shape[1]), f"Expected mu shape (1, {S.shape[1]}), but got {mu.shape}"
        self.testOK += 1
        self.testNum += 1
        assert sigma.shape == (1, S.shape[1], S.shape[1]), f"Expected sigma shape (1, {S.shape[1]}, {S.shape[1]}), but got {sigma.shape}"
        self.testOK += 1
        print("test_cluster_elements_no_split: OK")
        print("==============================================\n")

    def test_cluster_elements_split(self):
        S = np.random.rand(50, 3)
        w = np.ones(50) / 50.0
        min_var = 0.05
        mu, sigma = F.clusterElements(S, w, min_var)

        self.testNum += 1
        assert mu.shape[0] >= 1, f"Expected mu shape (N, {S.shape[1]}) with N >= 1, but got {mu.shape}"
        self.testOK += 1
        self.testNum += 1
        assert sigma.shape[0] >= 1, f"Expected sigma shape (N, {S.shape[1]}, {S.shape[1]}) with N >= 1, but got {sigma.shape}"
        self.testOK += 1
        print("test_cluster_elements_split: OK")

    def test_split(self):
        S = np.random.rand(50, 3)
        w = np.ones(50) / 50.0
        nodes = [F.Node(S, w)]

        new_nodes = F.split(nodes)
        assert len(new_nodes) == 2, f"Expected 2 nodes after splitting, but got {len(new_nodes)}"


def displayImage(title, img):
        plt.imshow(img)
        plt.title(title)
        plt.show()
        return title, img    

def main(img_name):
    a = testBeyasianMatting(img_name)
    a.testImgPath()
    a.testImageInfo()
    a.testConvertImage()
    a.test_fspecial()
    a.test_displayImage()
    a.test_node_creation()
    a.test_cluster_elements_no_split()
    a.test_cluster_elements_split()
    a.test_split()
    print("All passed")

if __name__ == '__main__':
    # img_name = input("Enter the name of the image file: ")
    img_name = 'GT01'
    main(img_name)
