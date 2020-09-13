<img src="https://github.com/jvhuang1786/teamHackTheBay/hackTheBay/blob/master/images/landcoverimage.png"></img>

# Analysis of Total Nitrogen Water Pollution Using Land and Air Data

## Team Hackathon Devpost Chesapeake Bay Water Quality Hackathon

## Introduction

This project was a team project that looked at land features, air quality and nitrogen oxide and its affect on the Chesapeake bay.  

The water quality data was collected from the hackathon github repo:

https://www.kaggle.com/splcher/animefacedataset

Landcover was collected through here:

http://mhxx-soubigazou.info

Air quality data was collected through here:

Nitrogen Oxide Data was collected through here:

**The goal of the model was to try to see what features and how it was affected **

### Libraries Used for the Hackathon


<table>

<tr>
  <td>Original Data Collection</td>
  <td>Nvidia GPU 11 gb and above, tensorflow-gpu 1.1.5, cuda 10.0, cuDNN 7.5, tensorrt 5.6.5</td>
</tr>

<tr>
  <td>Data Collection from other Sources</td>
  <td>numpy, os, cv2, glob, pandas, PIL, scipy, imageio, keras, augmentor, fastai</td>
</tr>

<tr>
  <td>Data Visualization</td>
  <td>keras.applications.vgg16, keras.preprocessing.image, matplotlib, numpy</td>
</tr>

<tr>
  <td>Modeling</td>
  <td>numpy, os, glob, imageio, time, PIL, keras, matplotlib</td>
</tr>

<tr>
  <td>Future Model</td>
  <td>numpy, scipy, keras, skimage.transform</td>
</tr>


</table>


## Original Data and merging it with landcover, air quality and nitrogen oxide. 

Main steps of in data Augmentation:

     Splitting the data in half
     Combining single image data to split data
     Mirroring the image data    
     Adjusting the brightness of the image data     
     Photoshop increase image resolution
     Resize the image to 512 x 512 and 256 x 256
     Run through fastai
     Use dataset_tool.py from Nvidia to transform to tfrecords


* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)


## Data Visualization and Findings

Visualization of Feature Map/Activation Map of images using VGG16 convolutional neural network

     Further from input the less details we can see.

* [Feature Map](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/feature_map.ipynb)

## Modeling 

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/mhxx.gif" width="480"></img>

Quick Visualization of image data using a DCGAN requires minimal computing power.

       Generator uses Conv2DTranspose
       Discriminator uses Conv2D
       Hyperparameters:
          Filter
          kernel_size
          Stride
          Padding
          kernel_initializer

* [Deep Convolutional GAN](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/dcgan_mhxx.ipynb)

## Future Models to Consider 

Best way to measure a GAN still is to look at fake images generated however there are quantitative measures.  Such as Frechet Inception Distance, Inception Score and Perceptual Path Length.

       FID measures the normal distribution distance between the real and fake images.  
       The closer the distance, the lower the score the better the image quality and diversity

* [Frechet Inception Distance](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/frechet_inception_distance.ipynb)



## Authors

* Justin Huang
* Tim Osburg
* Jen Wu
* Bryan Dickinson
* Berenice Dethier

