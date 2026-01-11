# IML_Music_Spotify_Project
Project for IML subject in PJAIT

# Dataset
[Spotify Track Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)

# Model Results
### Random Forest Results
```
============================================================
                   HYPERPARAMETER TUNING                    
============================================================
Best Parameters: 
- n_estimators: 300
- min_samples_split: 2
- min_samples_leaf: 2
- max_features: None
- max_depth: None

============================================================
                        TEST RESULTS                        
============================================================
- RMSE (Root Mean Sq. Error):    15.1706
- MAE  (Mean Absolute Error):    10.3634
- R2   (R-Squared Score):         0.5336

============================================================
                 TOP 10 FEATURE IMPORTANCE                  
============================================================
duration_ms          | ██                             0.0691
acousticness         | ██                             0.0688
speechiness          | ██                             0.0685
loudness             | ██                             0.0673
valence              | ██                             0.0672
danceability         | █                              0.0666
tempo                | █                              0.0665
energy               | █                              0.0638
liveness             | █                              0.0595
instrumentalness     | █                              0.0473
```

### Tensorflow Model
```
============================================================
               BEST HYPERPARAMETERS DETAILED                
============================================================
Learning Rate:     0.0005
Num Layers:        2.0000

	[Layer 1 INFORMATION]
Units:              64.0000
Dropout Enabled:     0.0000
Batch Norm:          1.0000

	[Layer 2 INFORMATION]
Units:              96.0000
Dropout Enabled:     1.0000
Dropout Rate:        0.1000
Batch Norm:          1.0000

============================================================
                        TEST RESULTS                        
============================================================
- RMSE (Root Mean Sq. Error):    19.1617
- MAE  (Mean Absolute Error):    14.1879
- R2   (R-Squared Score):         0.2560
```

### XGBoost
```
============================================================
                    BEST HYPERPARAMETERS                    
============================================================
- n_estimators: 504 
- max_depth: 10 
- learning_rate: 0.09247260686862264 
- subsample: 0.886456400096244 
- colsample_bytree: 0.9073751570616455

============================================================
                        TEST RESULTS                        
============================================================
- RMSE (Root Mean Sq. Error):    16.0190
- MAE  (Mean Absolute Error):    10.9180
- R2   (R-Squared Score):         0.4800
```