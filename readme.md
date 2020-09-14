<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/hack.png"></img>

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

Visualization of the Chemicals:

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/DO_Active_Chlorophyll.png" width="480"></img>

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/Nitrogen_Phosphorus.png" width="480"></img>

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/Suspended_Solids_Turbidity.png" width="480"></img>

  <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/pH_Temp_Salinity.png" width="480"></img>


     Further from input the less details we can see.
     

* [Chemical Visualization Notebook](https://github.com/teamHackTheBay/hackTheBay/blob/master/images/dfnitro_figures_2.ipynb)

    **Visualization of Location:**
    
    <img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/bryanloc.png" width="480"></img>

    
      * Visually from the sample locations that are highly correlated there seem to be many locations that align with Integration & Application Networks's most recent Chesapeake Bay Health (Total Nitrogen Threshold) Indicator Map (2013).
      * There a clumps of sample locations that were correlated with nearby NO2 air monitoring stations that also showed fair to poor on the 2013 Indicator map, including the Upper Bay (Upper Western Shore, Upper Eastern Shore, Patapsco and Back Rivers), Patuxent River, and Potomac River.
      * There mroe clusters of correlated sample locations further away from the bay, in New York, further up the Potomac and Susquehanna rivers.
      * These also seem to be clustered around cities, such as York, Lancaster, Charlottesville and others.
      * There does not seem to be many sites correlated with NO2 in the air in the Lower Easter Shore area of the Chesapeake Bay.
      * Most in/near the open water of the bay is not correlated with NO2 values
      * It appears, with the exception of New York, most of the sample locations that are near the open water of the bay are negatively correlated with the nearby monitoring station. And the positivly correlated sample sites are both near the open water of the bay and further away.
     
* [Correlation analysis on Nitrogen Oxide and location with Total Nitrogen](https://github.com/teamHackTheBay/hackTheBay/blob/master/exploration/epa_no2_monitoring_data.ipynb)

## Modeling 

**Xgboost Shap**

<img src="https://github.com/teamHackTheBay/hackTheBay/blob/master/images/xgboostshap.png" width="480"></img>

*Feature Importances and Hyperparameters used.* 

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

#Grid for hypertuning
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

      * The dependant variable (target) is the total nitrogen ('tn') in mg/L.
      
* [Notebook for Catboost model](https://github.com/teamHackTheBay/hackTheBay/blob/master/models/catboost/HackTheBay%20Catboost%20water_final%20dataset-no%20huc%20all%20features%20.ipynb)

## Chosen Model and Reasoning 

     Generator uses Conv2DTranspose
     Discriminator uses Conv2D
     Hyperparameters:
        Filter
        kernel_size
        Stride
        Padding
        kernel_initializer




## Future Models to Consider 

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/mhxx.gif" width="480"></img>

Feature Importances and Hyperparameters used. 

      Bryans Write up Section 

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

