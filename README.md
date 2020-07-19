# Hit Song Prediction
Code for practical experiments in Hit Song Prediction, including scripts that:
- Scrape a complete Billboard Hot 100 song dataset.
- Match Billboard songs with Spotify tracks.
- Extract audio features from a list of Spotify tracks.
- Train various learning algorithms on the task of Hit Song Prediction using Spotify's audio features.

## Results
These results were obtained with the hit songs from the Billboard Hot 100 charts from 2000 onward, and a random selection of songs from Spotify. Both csv files are in the `./datasets` directory. Results can differ when using different hit/non-hit datasets.

| Classifier | Accuracy | Precision | Recall | ROC-AP | ROC-AUC |
| --------- | --------- |  --------- |  --------- |  --------- |  --------- 
Logistic Regression | 0.819 | 0.783 | 0.880 | 0.832 | 0.877
Random Forest | 0.813 | 0.772 | 0.887 | 0.839 | 0.879
Neural network (MLP) | 0.818 | 0.784 | 0.876 | 0.834 | 0.877


<div align="center">
  <img width="50%" alt="CLMR model" src="https://github.com/Spijkervet/hit-song-prediction/blob/master/media/feature_importances.png?raw=true">
</div>
<div align="center">
  Feature importances extracted from Random Forest model
</div>


## Data collection
Two datasets are required to train: a "hit" and "non-hit" song dataset. The hit songs are scraped from the Billboard Hot 100 charts. Subsequently, the corresponding Spotify track is matched against the Billboard songs, and lastly their audio features are computed. While all files are already compiled in the `./datasets` folder, the following commands perform these operations:

```
# Downloads all songs from the Billboard Hot 100 chart:
python get_charts.py

# Matches the Billboard songs with Spotify track ID's using fuzzy string matching of the track and artist name:
python spotify_matcher.py

# Calculates audio features from Spotify on the matched tracks:
python spotify_features.py
```

Eventually, this yields a `spotify_billboard_features.csv` file containing all the Spotify features and Billboard data, which can be used as input for the learning algorithm.

## Usage
To start training using a hit- and non-hit song dataset in the `./datasets` folder, use:
```
python main.py
```

### Arguments
```
usage: main.py [-h] [--seed SEED] [--hits HITS] [--nonhits NONHITS]
               [--classifier {logistic_regression,random_forest,neural_network}]
               [--holdout_year HOLDOUT_YEAR] [--test_song TEST_SONG]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed
  --hits HITS           CSV file containing features of hit songs
  --nonhits NONHITS     CSV file containing features of non-hit songs
  --classifier {logistic_regression,random_forest,neural_network}
                        Classifier to use
  --holdout_year HOLDOUT_YEAR
                        Which year's hit songs to withold for testing
  --test_song TEST_SONG
                        Spotify URI to test hit song potential
```
