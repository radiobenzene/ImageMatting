# **Bayesian Matting**
The current project is a Python implementation of the paper published by Y.Y. Chuang, B. Curless, D. Salesin, R. Szeliski, A Bayesian Approach to Digital Matting.
Conference on Computer Vision and Pattern Recognition (CVPR), 2001.

# Group Details
* Group Name: Atomic Reactors
* Group Members and Tasks: 
  * Haohan Zhu -  QA, Testing, Interface Assembly
  * Alessandro Ferni - Quality Metrics, Image Compositing, Stress Testing
  * Uditangshu Aurangabadkar - Algorithmic Implementation, Stress Testing, UAT
# Implementation Specifics
## Pip Requirements
A list of all the required libraries can be found in the file `requirements.txt` or can be downloaded using the command
```
pip install -r requirements.txt
```

## Program Execution
To use the terminal version, type in the following command:
```
python main.py <image_name>
```
The images from the evaluation site alphamatting.com are saved as GT01 - GT27 in the `Images` directory. 

To use the interface version, type in the following command:
```
python engine.py
```
### Running the application
To start the application, type in the following command:
```
python engine.py -r
```
or 
```
python engine.py --run
```

# Algorithm Design
The algorithm design is based on the paper by Y. Chuang et al. The design can be shown using a flowchart.

# Quality Metrics
### Visualizing SAD values
The SAD values can be visualized using the following command:
```
python engine.py -s
```
or 
```
python engine.py --sad
```

### Visualizing MSE values
The MSE values can be visualized using the following command:
```
python engine.py -m
```
or 
```
python engine.py --mse
```

### Visualizing execution time
The execution time can be visualized using the following command:
```
python engine.py -t
```
or 
```
python engine.py --time
```

### Visualizing PSNR
The PSNR values can be visualized using the following command:
```
python engine.py -p
```
or 
```
python engine.py --psnr
```
# Testing
The testing of this application has been divided into 4 blocks.
## Unit Tests
* Testing if the image file from the path loader is an image object
* Testing if the trimap file from the path loader is an image object
* Testing if the size of the image and the trimap are of the same size
* Testing the correct conversion of the data type to double
* Testing the correct normalization of the data type to double
* Testing the shape of the Gaussian weighting kernel
* Testing the normalization of the kernel, i.e. the sum should be around 1
* Testing the `Node` function to check if the shape of the result matches the shape of the input
* Testing the `Shape` function of the `Node` class
* Testing the clustering algorithm with a large variance
* Testing the function `split` as part of the class `Node`
## E2E Tests
* The E2E test was designed in such that the user can select the image and trimap using an interface. To save the result with a new composite image, the user can select `save` to save the images
## Stress Tests
## Integration Tests
* Testing the Source Image Loader
* Testing the Trimap Path Loader
* Testing the number of channels in the source image - should be 3 (RGB)
* Testing the number of channels in the source trimap - should be 1 channel (Grayscale)
* Testing the `display()` function
* Testing if the displayed image matches the input image
* Testing the clustering algorithm on a normal variances
## UAT Tests
The UAT tests were built using Atlassian and Jira and can be found on https://atomicreactors.atlassian.net/l/cp/yoUxiYWD 

# Acknowledgments
The group would like to thank the following members of the community:
* Ronald Laban - For being the group's consultant regarding the algorithm design
* Michael Rubinstein's lecture - For guiding the implementation of the application\
https://people.csail.mit.edu/mrub/bayesmat/index.html
* Marco Forte - For the author's implementation of the Orchard-Boumann clustering algorithm
https://github.com/MarcoForte
* GPT-3 - For providing improvement suggestions to the Python implementation
https://chat.openai.com/


