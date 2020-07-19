import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

if __name__ == "__main__":
    TRACKS_PER_REQUEST = 50  # Spotify yields up to 50 audio features per request

    df = pd.read_csv("datasets/spotify_billboard.csv")
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

    track_ids = df["id"]
    chunks = [
        track_ids[x : x + TRACKS_PER_REQUEST]
        for x in range(0, len(track_ids), TRACKS_PER_REQUEST)
    ]
    spotify_features_df = pd.DataFrame()

    try:
        for track_ids in tqdm(chunks):
            features = sp.audio_features(track_ids)

            for tid, feature in zip(track_ids, features):
                del feature["type"]
                d = df[df["id"] == tid].to_dict("r")[0]
                d = {**d, **feature}
                spotify_features_df = spotify_features_df.append(d, ignore_index=True)
    except Exception as e:
        print(e)
        pass

    spotify_features_df.to_csv("datasets/spotify_billboard_features.csv", index=False)
