# IML_Music_Spotify_Project
Project for IML subject in PJAIT

# Dataset
[Spotify Track Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)

# Model Results
### Random Forest Results
```
============================================================
                        TEST RESULTS                        
============================================================
- RMSE (Root Mean Sq. Error):    15.2850
- MAE  (Mean Absolute Error):    10.3182
- R2   (R-Squared Score):         0.5266

============================================================
                 TOP 10 FEATURE IMPORTANCE                  
============================================================
speechiness          | ██                             0.0683
duration_ms          | ██                             0.0681
acousticness         | ██                             0.0681
valence              | ██                             0.0667
loudness             | █                              0.0666
tempo                | █                              0.0665
danceability         | █                              0.0664
energy               | █                              0.0631
liveness             | █                              0.0591
instrumentalness     | █                              0.0469
```

### Tensorflow Model
```
============================================================
                        TEST RESULTS                        
============================================================
- RMSE (Root Mean Sq. Error):    19.1530
- MAE  (Mean Absolute Error):    14.2915
- R2   (R-Squared Score):         0.2566
```