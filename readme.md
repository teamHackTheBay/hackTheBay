<img src="https://github.com/jvhuang1786/teamHackTheBay/hackTheBay/blob/master/images/landcoverimage.png"></img>

# Analysis of Total Nitrogen Water Pollution Using Land and Air Data

## Team Hackathon Devpost Chesapeake Bay Water Quality Hackathon

## Introduction

This project was a team project that looked at land features, air quality and nitrogen oxide and its affect on the Chesapeake bay.  

The water quality data was collected from the hackathon github repo:

https://drive.google.com/file/d/12uoFlcn8pgeuxD2-seFak36KTvrFPKCt/view

Landcover was collected through here:

https://www.mrlc.gov/viewer/

Air quality data was collected through here:

https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-regional-reanalysis-narr

Nitrogen Oxide Data was collected through here:

https://echo.epa.gov/tools/data-downloads

**The goal of the model was to try to see what features and how it was affected**

### Libraries Used for the Hackathon


<table>

<tr>
  <td>Original Data Collection</td>
  <td>pandas, matplotlib, numpy regex</td>
</tr>

<tr>
  <td>Data Collection from other Sources</td>
  <td>pandas, geopy, numpy, seaborn, sklearn, geopandas, certify, urllib3, scipy.spatial</td>
</tr>

<tr>
  <td>Data Visualization</td>
  <td>plotly, seaborn, matplotlib</td>
</tr>

<tr>
  <td>Modeling</td>
  <td>catboost, randomforest from sklearn, xgboost, shap</td>
</tr>

<tr>
  <td>Future Model</td>
  <td>sklearn random forest, shap</td>
</tr>


</table>


## Original Data and merging it with landcover, air quality and nitrogen oxide. 

Steps of Collected the Data: 

     Total Nitrogen was first collected out of the water_final.csv from the hackathon repo.
     Landcover was collected 
     Narr air quality data was collected    
     These features were merged together with the total nitrogen chemical choosing total nitrogen from the parameter feature in water_final.csv
     Nitrogen oxide data and correlation was added into the csv. 
     Since land coverage data was from 2016 we decided to use datapoints from 2016 to the end of 2019. 
     A chemicals csv was also created to take a look at the relationship among chemicals in the water. 

* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)
* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)
* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)


## Data Visualization and Findings

Visualization of the Chemicals:

     Further from input the less details we can see.
     

* [Feature Map](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/feature_map.ipynb)

Visualization of Location:

     Further from input the less details we can see.
     
* [Feature Map](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/feature_map.ipynb)

## Modeling 

Shap file here 

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/mhxx.gif" width="480"></img>

Feature Importances and Hyperparameters used. 

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

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/mhxx.gif" width="480"></img>

Feature Importances and Hyperparameters used. 

       Generator uses Conv2DTranspose
       Discriminator uses Conv2D
       Hyperparameters:
          Filter
          kernel_size
          Stride
          Padding
          kernel_initializer

* [Deep Convolutional GAN](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/dcgan_mhxx.ipynb)



## Authors

* Berenice Dethier
  Science, travel, and food enthusiast
  [GitHub: berenice-d](https://github.com/berenice-d)
* Bryan Dickinson
  [GitHub: bryan-md](https://github.com/bryan-md)
* Justin Huang
  Into anime, finance computer vision, and NLP.
  [GitHub: Jvhuang1786](https://jvhuang1786.github.io/)
* Tim Osburg
  Geophysicist and avid runner.
  [GitHub: Tosburg](github.com/Tosburg)
* Jen Wu
  Entrepreneur and nature nerd.
  [GitHub: Jenxwu](https://github.com/Jenxwu) (edited) 

