[![Watch the hackthebay submission](https://github.com/teamHackTheBay/hackTheBay/blob/master/images/hack.png)](https://www.youtube.com/watch?v=kAa5iWRKkNc&feature=youtu.be)
### Click on the image to watch the video presentation on youtube.

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

     * Total Nitrogen was first collected out of the water_final.csv from the hackathon repo.
     * Landcover was collected 
     * Narr air quality data was collected    
     * These features were merged together with the total nitrogen chemical choosing total nitrogen from the parameter feature in water_final.csv
     * Nitrogen oxide data and correlation was added into the csv. 
     * Since land coverage data was from 2016 we decided to use datapoints from 2016 to the end of 2019. 
     * A chemicals csv was also created to take a look at the relationship among chemicals in the water. 

**CSV Files and data collection write up**
* [Detailed Description of the Steps](https://github.com/teamHackTheBay/hackTheBay/blob/master/writeup-sections/data_acquisition_writeup.pdf)
* [Final CSV using Landcover, airquality and nitrogen oxide](https://github.com/teamHackTheBay/hackTheBay/blob/master/data/final_water.csv)
* [Chemical Comparison CSV](https://github.com/teamHackTheBay/hackTheBay/blob/master/images/dfnitro.csv)

**Feature engineering and merging**
* [Adding of Distance Feature](https://github.com/teamHackTheBay/hackTheBay/blob/master/feature_engineering/add_distance_feat.ipynb)
* [Adding huc12_enc Feature](https://github.com/teamHackTheBay/hackTheBay/blob/master/feature_engineering/huc_mean_target.ipynb)
* [Importing Narr Data](https://github.com/teamHackTheBay/hackTheBay/blob/master/exploration/narr_import.ipynb)
* [Joining LandCover and Narr with hackthebay csv](https://github.com/teamHackTheBay/hackTheBay/blob/master/exploration/htb_lc_data_join.ipynb)
* [Gathering Nitro Oxide data](https://github.com/teamHackTheBay/hackTheBay/blob/master/exploration/epa_no2_monitoring_data.ipynb)
* [Combining new features with Total Nitrogen](https://github.com/teamHackTheBay/hackTheBay/blob/master/exploration/adding_nitro_oxide.ipynb)


## Data Visualization and Findings

**Visualization of the Chemicals:**

Scatter plot Dissolved Oxygen and Chlorophyll-A

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/DO_Active_Chlorophyll.png" width="480"></img>
  
Scatter plot Total Nitrogen and Phosphorus

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/Nitrogen_Phosphorus.png" width="480"></img>

Scatter plot Suspended Solids and Turbidity 

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/Suspended_Solids_Turbidity.png" width="480"></img>
  
Distribution of PH, Salinity and Water Temperature 

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/pH_Temp_Salinity.png" width="480"></img>
  
Temperature Map in regards to Chesapeake Bay 
  
  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/temp.gif" width="480"></img>
     
* [Chemical Visualization Notebook](https://github.com/teamHackTheBay/hackTheBay/blob/master/images/dfnitro_figures_2.ipynb)

    **Visualization of Location:**
    
    <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/bryanloc.png" width="480"></img>

    
      * Visually from the sample locations that are highly correlated there seem to be many locations that align with Integration & Application Networks most recent Chesapeake Bay Health (Total Nitrogen Threshold) Indicator Map (2013).
      * There a clumps of sample locations that were correlated with nearby NO2 air monitoring stations that also showed fair to poor on the 2013 Indicator map, including the Upper Bay (Upper Western Shore, Upper Eastern Shore, Patapsco and Back Rivers), Patuxent River, and Potomac River.
      * There more clusters of correlated sample locations further away from the bay, in New York, further up the Potomac and Susquehanna rivers.
      * These also seem to be clustered around cities, such as York, Lancaster, Charlottesville and others.
      * There does not seem to be many sites correlated with NO2 in the air in the Lower Easter Shore area of the Chesapeake Bay.
      * Most in/near the open water of the bay is not correlated with NO2 values
      * It appears, with the exception of New York, most of the sample locations that are near the open water of the bay are negatively correlated with the nearby monitoring station. And the positively correlated sample sites are both near the open water of the bay and further away.
     
* [Correlation analysis on Nitrogen Oxide and location with Total Nitrogen](https://github.com/teamHackTheBay/hackTheBay/blob/master/exploration/epa_no2_monitoring_data.ipynb)

## Modeling 

**Xgboost Shap**

<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/xgboostshap.png" width="480"></img>

*Feature Importance and Hyperparameters used.* 

Robust Scaler using knn imputer of two nearest neighbors with the following features:

```python
#robust scaler column transformer 
ct = make_column_transformer(
(knn_impute_scale1, ['areaacres', 'lc_21', 'lc_31', 'lc_41', 'lc_42', 'lc_43', 'lc_52',
   'lc_71', 'lc_81', 'lc_82', 'lc_90', 'lc_95', 'year', 'week',
   'airtemp_narr', 'precip3_narr', 'humidity_narr', 'cl_cover_narr',
   'sfc_runoff', 'windspeed_narr', 'wdirection_narr', 'precip48_narr',
   'of_dist', 'total Nitrogen Oxide in year']),
remainder='passthrough')
```

Hyperparameter Tuning:

```python
#Set Parameters
params = {}
params['xgbregressor__max_depth'] = np.arange(3,11,1)
params['xgbregressor__n_estimators'] = np.arange(0, 1000,10)
params['xgbregressor__min_child_weight'] = np.arange(1,11,1)
params['xgbregressor__importance_type'] = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

#Grid for hyper tuning
gridRF = RandomizedSearchCV(pipeline, params, cv = 5, n_jobs = -1, random_state = 0, n_iter = 30, scoring = 'r2')
```

* [Notebook for chosen Xgboost model](https://github.com/teamHackTheBay/hackTheBay/blob/master/models/all_feature_model/no_huc_xgb/all_features_no_huc_corr_xgb_model.ipynb)

**Catboost Shap**

<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/catboostshap.png"></img>

*Feature Importances and Hyperparameters used.* 

    In this notebook, we use the following variables for prediction: 
       
       * 'latitude', 'longitude', 'areaacres', 'za_mean', ('lc_21', 'lc_22', 'lc_23', 'lc_24') combined as lc_2t, 'lc_31', ('lc_41', 'lc_42', 'lc_43') combined as lc_4t, 'lc_52', 'lc_71', 'lc_81', 'lc_82', ('lc_90', 'lc_95') combined as lc_9t, month', 'year', 'week', 'dayofweek', 'hour', 'min', 'quarter', 'airtemp_narr', 'precip3_narr', 'humidity_narr', 'cl_cover_narr', 'sfc_runoff', 'windspeed_narr', 'wdirection_narr', 'precip24_narr', 'precip48_narr', 'of_dist', 'total Nitrogen Oxide in year', and 'date_delta'.
       
      * Date_delta is a numeric variable which capture the time in seconds from the latest record. 
      
      * We could not keep 'new_date' in a datetime format (not supported by Catboost). 
      
      * The reasoning behind creating date_delta is that other time variables (month, year, week, day of week and quarter) are categorical. 
      
      * They can capture a seasonal phenomenon (pollution from industry on weekdays for example) but not a trend over time.

      * We removed the following variables: 'new_date' (replaced by datedelta which is numeric), 'huc12', and 'huc12_enc'.

      * The dependent variable (target) is the total nitrogen ('tn') in mg/L.
      
* [Notebook for Catboost model](https://github.com/teamHackTheBay/hackTheBay/blob/master/models/catboost/HackTheBay%20Catboost%20water_final%20dataset-no%20huc%20all%20features%20.ipynb)

## Chosen Model and Reasoning 

     Catboost model did the best job of explaining total nitrogen across the Chesapeake Bay.  
      * More Farmland lc_82 the higher the total nitrogen
      * The higher the latitude the more total nitrogen this was similar to the feature of_distance
      * The lower the nitro oxide the lower the total nitrogen
      
```python
#R Squared
print(r2_score(y_test, y_pred))
0.8401995362308845

#Explained Variance Score
print(explained_variance_score(y_test, y_pred))
0.8402718767819098

#Root Mean Squared Error
print(np.sqrt(mean_squared_error(y_test, y_pred)))
0.9249663789834368
```
      
## A further model improvement 
* [Random Forest Ensemble Model Notebook](https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/notebook/1-ensemble_model.ipynb)
#### During data analysis we found that feature relationships with TN varied within the watershed. 

```python
Ensemble Model Evaluation Metrics
Mean Squared Error: 0.37866608251212613
Root Mean Square Error: 0.6153584991792396


```


<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/distance_tn.png" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/indicator.png" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/fi_model1.png" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/model1_shapsum.png" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/fi_model2.png" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/model2_shapsum.png" width="450"></img>

      * Further segmentation or additional relevant features may help make more accurate predictions. 
      
      * When looking at the feature importance side by side, the target mean-encoding of the HUC12 feature is the most important feature for both models. 
      * This makes sense that the previous averages of TN within the HUC will help predict future TN readings, this feature helps capture the variability between HUC12 areas.
      
      * The first group's top features aside from the mean encoding feature:
          * month
          * air temperature
          * rainfall in the past 24 hours
          * NO2 emissions from point sources
          * humidity
          * mean value of air NO2 and the year
      * The second group's top features aside from the mean encoding feature:
          * lc_82(ratio of land that is 'cultivated crops')
          * distance from the bay
          * mean value of air NO2
          * rainfall in the past 24 hours
          
      * This shows that there is a difference in the relationships of variables and TN sampled in the water. 
      * It would seem that Group 1's TN values rely upon seasonal fluctuations, weather, NO2 emissions (from correlated point sources) and air NO2 values (from point sources and nearby cities/non-point sources). 
      
      * This could mean that a focus on reducing TN from point sources and non-point sources would help reduce TN in the bay.
      * Group 2's TN values rely more upon how land cover is utilized, specifically crop land. Determining ways to mitigate cropland run off could help reduce TN.      
      * An explanation for 'distance to the outflow of the bay' being such an important feature, is that there is less water for the pollutant to be diluted in the further from the bay you are, making run off TN values



**Improving the above model when it comes to scalability.** <br>
The above model requires HUC12 boundary's history of Total Nitrogen samples. We could use a global mean for that portion of the data, however, the model can be further improved by removing this mean encoded feature. When removing this feature we see a further reduction in RMSE.
* [Ensemble Model - mean encoding removed Notebook](https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/notebook/2-%20ensemble_model-mean_encoding_removed.ipynb)
```python
Ensemble Model Evaluation Metrics
Mean Squared Error: 0.340
Root Mean Square Error: 0.583
Mean Abs. % Error: 19.917
```
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/fi2_model1.PNG" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/fi2_model1_shap.PNG" width="450"></img>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/fi2_model2.PNG" width="450"></img><img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/fi2_model2_shap.PNG" width="450"></img><br>
While there is an observed improvement from the XGBoost or Catboost model, there is still room for improvement, as the mean % absolute error is ~20%. Plotting these will reveal that the residuals are larger for larger values of TN.

For the model with observations of larger TN and further from the mouth of the bay one of the most important features is *lc_82* which corresponds to 'cultivated crops', however this doesn't give the best indication of which crops are impacting TN in the watershed. You can see in the difference between SHAP plots of the Catboost model that indicate a higher % of an HUC area leads to higher TN values, and the hurdle ensemble model that show higher percent area of HUC lead to lower TN values. There are more than 70 categories of crops in the Chesapeake Bay watershed, the most in terms of acreage being hay, corn & soybeans for 2019. These crops may impact TN in the watershed differently, as they have different nitrogen updatakes.  <br/>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/crops_capture_ws.PNG" title="Land Cover Category Acreage in the Watershed" alt="https://nassgeodata.gmu.edu/CropScape/"></img>

**Utilizing crop features** <br>
Lastly, a model was built for portion of observations that had a high TN. The land cover features from all previous models were replaced with land use data describing the type of crop, along with previous categories (developed land, forest etc.) yearly from 2015-2019. These values were not normalized per observation, equating to roughly 1 value (or pixel value) equates to 30 meters or land cover type<sup>1</sup>. A histogram-based gradient boosting regression tree was used to model the data with a poisson loss.
* [Ensemble Model - crop_data Notebook](https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/notebook/3-ensemble_model-crop_data.ipynb)

The data was obtained from USDA's Cropscape.
> **What data is hosted on the CropScape website?**

>The geospatial data product called the Cropland Data Layer (CDL) is hosted on  [CropScape](https://nassgeodata.gmu.edu/CropScape/). The CDL is a raster, geo-referenced, crop-specific land cover data layer created annually for the continental United States using moderate resolution satellite imagery and extensive agricultural ground truth. All historical CDL products are available for use and free for download through CropScape
><font size=2>https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php</font>


 Distance feature was also removed. This model will predict TN of a HUC12 boundary with area of crops, weather data, nearby NO2 monitoring data, and emissions from correlated NO2 emissions from point source locations within the airshed. 

```python
Model Evaluation Metrics
Mean Squared Error: 1.7
Root Mean Square Error: 1.3
Mean Abs. % Error: 18.7
```
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/crops_perm_importance.PNG" width="500"></img>
<br>
<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/crops_shap.PNG"></img>

This model, while it has a higher rmse of **1.3** from **.58** from the model that uses *lc_82*, it has a lower mean absolute % error and gives some insight into how crops impact TN. 

When looking at the feature importances from these plots, it appears the top 5 important features are:
* corn
* open water
* alfalfa
* deciduous forest
* double crop barley/soybeans
>Double Cropping
>is the practice of growing two or more crops in the same piece of land during one growing season  instead of just one crop. <font size=2>https://en.wikipedia.org/wiki/Multiple_cropping</font>

Open water & Deciduous Forest have an inverse correlation to TN. This could be that a reduction of the area of these two, mean more developed land (for people, crops, pastures etc.) that contribue to nitrogen runoff. It could also be from the dilution of the pollutant (more open water) or some environmental process of deciduous forests on nitrogen runoff.

Clovers, double crops of barley/soybeans and corn seem to have the highest impact to TN in the watershed.  Something to keep in mind about these results. Corn and soybeans both require a lot of nitrogen. Corn utilizes 1lb of nitrogen / bushel of grain<sup>2</sup>, and soybean utilizes 5lbs of n/bushel.<sup>3</sup>. Corn and soybeans are also among the top 10 agricultural exports or higher for most states in the watershed.  For Maryland, soybeans ranks 3rd and corn ranks 4th<sup>4</sup>.   When looking at correlations of crop area to TN, alfalfa, grassland/pasture & pumpkins round out the top 5 crops. 

<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/models/ensemble_model/visuals/crops_corr.PNG" title="Top Correlated Features with TN"></img>

**More study is needed** <br>
The above importance features showcase the complexity of this problem. For example, an interesting crop shown above as impactful is 'clover'.  Clovers <sup>5</sup>(and alfalfa<sup>6</sup>) can be utilized to increase the soil's nitrogen levels, taken from the air. So while it may look like it would be beneficial to not plant clovers higher amounts of clovers seem to lead to higher concentrations of TN in the tributaries, utilizing clovers could have an overall positive effect on the environment<sup>7,8</sup>.  
Another interesting observation, from the plots above, it shows that wetlands (*lc_90*, *lc_95*, *lc_9t*) show a somewhat inverse correlation to TN. However alfalfa needs dry roots to be effective, alfalfa grown in wetlands will not be as affective in fixating nitrogen<sup>9</sup>. 

* Corn is one of the leading crops in the area and requires the most nitrogen. Focusing on nitrogen management on corn fields could help reduce TN found in the bay.<sup>10</sup>
* Looking at distributions of TN across these crops could help give some insight into what runoff might look like and how to minimize the flow of nitrogen in the tributatries and the bay.
* More knowledge of nitrogen utilization for these crops, the amounts needed, and application may prove useful and trying to determine how to limit TN runoff into the Chesapeake bay.
* Using a hurdle poisson regression model <sup>11</sup>could prove useful in helping to predict and discover more details about the different factors contributing to TN in the bay.
* Further exploration segmenting different HUC12 boundaries in the watershed could prove useful in identifying the different distrubutions of TN in the bay
* Consider adjusting report card/indicator metrics to take into account how close/far from the mouth of the bay <sup>12</sup>.  Sites near the bay are seemingly 'good' or 'very good', however the pollution at those points may be diluted and therefore understating the amount of TN  being realsed into the bay compared to HUC areas upstream.



 <font size=2><sup>1</sup>https://www.nass.usda.gov/Research_and_Science/Cropland/docs/MuellerICASVI_CDL.pdf <br>
<sup>2</sup>https://emergence.fbn.com/agronomy/how-much-nitrogen-does-your-corn-need <br>
<sup>3</sup> https://agfax.com/2014/01/02/adding-nitrogen-soybeans-can-improve-yields/ <br>
<sup>4</sup>https://www.ers.usda.gov/data-products/state-export-data/annual-state-agricultural-exports/<br>
<sup>5</sup>https://www.uaex.edu/publications/pdf/FSA-2160.pdf <br>
<sup>6</sup>http://www.midwestforage.org/pdf/61.pdf.pdf <br>
<sup>7</sup>https://news.stanford.edu/news/2010/february22/legumes-nitrogen-fertilizer-022610.html<br>
<sup>8</sup>https://www.resilience.org/stories/2018-09-20/improving-air-quality-with-clover/<br>
<sup>9</sup>https://ucanr.edu/blogs/blogcore/postdetail.cfm?postnum=10478 <br>
<sup>10</sup>https://www.ers.usda.gov/amber-waves/2011/september/nitrogen-footprint/
<sup>11</sup>https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013WR014372 <br>
<sup>12</sup>https://ian.umces.edu/ecocheck/report-cards/chesapeake-bay/2012/indicators/total_nitrogen/#_Data_Map <br></font>
## Authors

* Berenice Dethier
  Science, travel, and food enthusiast
  [GitHub: berenice-d](https://github.com/berenice-d)
* Bryan Dickinson
  Curious. Addicted to Coffee.
  [GitHub: bryan-md](https://github.com/bryan-md)
* Justin Huang
  Into anime, finance computer vision, and NLP.
  [GitHub: Jvhuang1786](https://jvhuang1786.github.io/)
* Tim Osburg
  Geophysicist and avid runner.
  [GitHub: Tosburg](github.com/Tosburg)
* Jen Wu
  Entrepreneur and nature nerd.
  [GitHub: Jenxwu](https://github.com/Jenxwu) 



