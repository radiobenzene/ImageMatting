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
    options = "rmsthu"
    
    # Creating a dictionary of options
    long_options = ["run", "mse", "sad", "time", "help", "unit"]
    
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
                displaySAD('SAD.txt')
                
    
    except getopt.err as error:
        err = "Invalid Option"
        print(str(err))
    

#main(user_selector = True, save = True, show_FG = True, compositing= True)
