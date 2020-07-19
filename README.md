# hit-song-prediction
Hit Song Prediction

## Data collection
The dataset is divided between "hit" songs and "non-hit" songs. The hit songs are scraped from the Billboard Hot 100 charts. Then, the corresponding Spotify track is matched against the Billboard Hot 100 songs, and lastly their audio features are computed. The following commands perform these operations:

```
# Downloads all songs from the Billboard Hot 100 chart:
python get_charts.py

# Matches the Billboard songs with Spotify track ID's using fuzzy string matching of the track and artist name:
python spotify_matcher.py

# Calculates audio features from Spotify on the matched tracks:
python spotify_features.py
```

Eventually, this yields a `spotify_billboard_features.csv` file containing all the Spotify features and Billboard data, which can be used as input for the learning algorithm.

## Results
