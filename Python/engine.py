from cv2 import *
from functions2 import *
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import getopt
import testFlow
from metrics import *
# Main Engine


# n = len(sys.argv)
# image_name = sys.argv[1]
if __name__ == "__main__":
    argument_list = sys.argv[1:]
    
    # Condensed options
    options = "rmsthp"
    
    # Creating a dictionary of options
    long_options = ["run", "mse", "sad", "time", "help", "psnr"]
    
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argument_list, options, long_options)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-r", "--run"):
                print("Running the application")
                main(user_selector = True, save = True, show_FG = True, compositing= True)
                break
            elif currentArgument in ("-s", "--sad"):
                print("Showing SAD values")
                displayMetric('SAD.txt', 'SAD Values', 'Image Name', 'SAD Values')
            elif currentArgument in ("-m", "--mse"):
                print("Showing MSE values")
                displayMetric('mse.txt', 'MSE', 'Image Name', 'MSE Value')
            elif currentArgument in ("-t", "--time"):
                print("Showing Execution Times")
                displayMetric('execution.txt', 'Execution Time', 'Image Name', 'Execution Time (in seconds)')
            elif currentArgument in ("-p", "--psnr"):
                print("Showing PSNR")
                displayMetric('psnr.txt', 'PSNR', 'Image Name', 'PSNR')
            elif currentArgument in ("-h", "--help"):
                print("Showing Help")
                print("-h or --help for help")
                print("-r or --run to run")
                print("-m or --mse to display MSE")
                print("-s or --sad to display SAD")
                print("-p or --psnr to display PSNR")
                print("-t or --time to display execution time")

    except getopt.err as error:
        err = "Invalid Option"
        print(str(err))
    

#main(user_selector = True, save = True, show_FG = True, compositing= True)
