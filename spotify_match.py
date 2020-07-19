import pandas as pd
import re
from fuzzywuzzy import fuzz
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

def remove_bracketed_text(text):
    return re.sub("[\(\[].*?[\)\]]", "", text)


if __name__ == "__main__":

    df = pd.read_csv("billboard.csv")
    df["SongID"] = df["title"] + df["artist"]
    df = df.drop_duplicates(subset="SongID", keep="first")

    spotify_df = pd.DataFrame(columns=[])
    unmatched_df = pd.DataFrame(columns=[])
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        matched = False
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
        results = sp.search(row.title, type="track")
        for r in results["tracks"]["items"]:
            artists = r["artists"]
            popularity = r["popularity"]
            name = r["name"]
            artist_names = ", ".join([a["name"] for a in artists])

            title_ratio = fuzz.ratio(remove_bracketed_text(row.title.lower()), remove_bracketed_text(name.lower()))
            artist_ratio = fuzz.ratio(remove_bracketed_text(row.artist.lower().replace("featuring", "")), remove_bracketed_text(artist_names.lower()))

            if title_ratio > 90 and artist_ratio > 80:
                d = {
                    "title_ratio": title_ratio,
                    "artist_ratio": artist_ratio,
                    "popularity": popularity,
                    "name": name,
                    "artists": artist_names,
                    "id": r["id"],
                    "uri": r["uri"],
                    "album": r["album"]["name"],
                    "preview_url": r["preview_url"]
                }
                spotify_df = spotify_df.append(d, ignore_index=True)
                matched = True
        
        if not matched:
            unmatched_df = unmatched_df.append(row)

    print("Unmatched tracks:", len(unmatched_df))
    spotify_df.to_csv("spotify_billboard.csv")
    unmatched_df.to_csv("unmatched_spotify_billboard.csv")
