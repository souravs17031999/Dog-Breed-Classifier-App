# Project Objective : Write an Algorithm for a Dog Identification App using Deep Learning    
# Dataset :    
[Click here for Dataset1](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) [Click here for Dataset2](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).     
# Motivation :   
learning and understanding of Convolutional Neural Networks    

# Pipeline for Project :   
* First thing in this pipeline is recognition algorithm for both humans and dogs and then classify it by giving out the exact name of breed.    
![p1.png](/Images/p1.png)   

# Human Face detector :    
```
We can try different pretrained algorithms by OpenCV.   
I have tried HOG, LBP, HAAR etc..   or any other deep learning based pre trained models.      
```
* [HAAR](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml):  
This algorithm is also called voila jones algorithm based on HAAR like wavelets.
HAAR wavelets are sequence of rescaled square shaped functions which is explained in detailed way [here](https://en.wikipedia.org/wiki/Haar_wavelet).     

![p3.png](/Images/p3.png)            

HAAR like features for detection :      

![p2.png](/Images/p2.JPG)        

A target window of some determined size is moved over the entire input image for all possible locations to calculate HAAR like features and since it was a very high computational task, therefore alternative method using *integral images* was designed.
The way it works is described briefly below : 

Integral images calculation reduced the computations :    

In below figure, haar works by calculating the difference between sum of black and sum white shades and let's say here it comes out to be something like :   
&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; haar feature &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;real images      
            
![p4.png](/Images/p4.png)        
![p5.jpg](/Images/p5.JPG)         

The closer this difference is to "1", then most probably a haar feature has been detected !     

Integral images :    
![p6.jpg](/Images/p6.JPG)         

* [HOG](https://github.com/opencv/opencv/blob/master/data/hogcascades/hogcascade_pedestrians.xml):       

Histogram of oriented gradients is calculated by taking difference in pixel intensities for every block of pixel in a 64 * 64 window, similar to sliding window over the entire image.    
This is based on the fact that, certain regions of our face have slightly darker shades over the other and thus there becomes gradient oientation of vector in some localized portions of our face.

Like in this image, we can see the gradient magnitude and gradient direction:      

![p7.jpg](/Images/p7.JPG)           

Now calculating for all pixel blocks:     

![p8.jpg](/Images/p8.JPG)            

For more detailed explanation, [click here](https://medium.com/analytics-vidhya/a-take-on-h-o-g-feature-descriptor-e839ebba1e52).

* [LBP](https://github.com/opencv/opencv/blob/master/data/lbpcascades/lbpcascade_frontalface.xml):             

Local binary patterns is a algorithm for feature detection based on local representation of texture.      
How it's calculated ? Let's see...       
For every block (in grayscale) , we select a center pixel value and construct a threshold by indicating 1 if value in center is greater than or equal to neighbouring one otherwise zero and then construct a 1 -D array by warping around either in clockwise or anticlockwise direction.    
(Here i show for one of the central pixel - "10", but it is done for every other pixel block)    

![p9.png](/Images/p9.png)            

Then, a histogram of 256 bin is constructed from the final output lbp pattern image.   

![p10.png](/Images/p10.png)                

# Dog Face detector :       

* Here also, we can use same above detectors.        

### Some examples :     
![p11.JPG](/Images/p11.JPG)                      

# CNN classification :           

* Now, that we have recognized that if image contains a dog face, a human face or none of them.    
It's time for training our own neural network for classifiying the breed of dog if image contains dog (or most resembled label for human !)   
So, let's get started.....     

```
We can this here using two different approaches :    
* Constructing CNN from scratch    
* Using pre trained CNN models 
```    
#### CNN from scratch and it's overview :      

```
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))        
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))      
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))     
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)         
  (fc1): Linear(in_features=50176, out_features=500, bias=True)     
  (fc2): Linear(in_features=500, out_features=133, bias=True)      
  (dropout): Dropout(p=0.5, inplace=False)       
```

#### Pre trained [ResNet50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model :    
(Reasons of choosing this model has been included in the notebook itself)     
* This architechture contains **_(conv1)_** as first convolutional layer containing in channels as 3 which is due to **_RGB input tensor_** , **_(bn1)_** as batch normalization layer, followed by **_ReLU_** and **_MaxPooling_** and then it contains 4 main layers named **_layer1_**, **_layer2_**, **_layer3_** and **_layer4_** which contains further sub layers of convolution followed by **_batchnorm_** followed by **_relu_** followed by **_maxpooling_** , and then finally fc.   
* **ReLU** activation is used as it's the most proven activation function for classification problems as it introduces good and right amount of non linearity with less chances of vanishing gradient problem !        
* **Batch normalization** helped in making the network more stable and learning faster thereby faster convergence.     
* **Maxpooling** helped in downsampling high number of parameters created by producing higher dimensional feature maps after convolution operation and thus selecting only relevant features from the high dimensioned feature matrix.       
* Then i replaced last layer of this architechture by fully connected layer containing two sub linear layers as follows :   ```Linear(in_features=2048, out_features=512) Linear(in_features=512, out_features=133)```   
with ReLU activations between the linears.        

![p12.png](/Images/p12.png)                       
# Some graphics of data augmentation used :      
* Augmentation used :      
```   
transforms.RandomRotation(10),       
transforms.RandomResizedCrop(224),      
transforms.RandomHorizontalFlip()     
```      
![p13.JPG](/Images/p13.JPG)                         

# Finally some examples/results :               


# Getting started :     
* For getting started locally on your own system, [click here](#).

# Navigating Project : 
* [Check out the complete source code including training and testing codes](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/dog_app.ipynb)     
* [If you just want the raw jupyter notebook, check out report here](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/dog_app.html)        
* [For checking deep inside model parameters and shapes, click here](https://github.com/souravs17031999/Dog-Breed-Classifier-App/blob/master/summary.txt)      
* [Want sample images for testing, download here](#)    
* [Want pre trained model weights, download here](#)    

# Links to references for more detailed learning :     
* [Face recognition](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
* [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
* [Gradient descent](https://medium.com/secure-and-private-ai-writing-challenge/playing-with-gradient-descent-intuition-e5bde385078?source=---------11------------------)
* [Backpropogation](https://medium.com/secure-and-private-ai-writing-challenge/playing-with-backpropagation-algorithm-intuition-10c42578a8e8?source=---------10------------------)
* [Data augmentation](https://medium.com/secure-and-private-ai-writing-challenge/data-augmentation-increases-accuracy-of-your-model-but-how-aa1913468722?source=---------6------------------)
* [Pytorch docs](https://pytorch.org/docs/stable/index.html)
* [Udacity course Deep learning ND]()     
       
    
⭐️ this Project if you liked it !
