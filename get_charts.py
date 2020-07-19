import os
import billboard
import pandas as pd
from datetime import datetime, timedelta

def get_charts(date):
    chart = billboard.ChartData('hot-100', date=date)
    while chart.previousDate:
        chart = billboard.ChartData('hot-100', chart.previousDate)
        for c in chart:
            song = vars(c)
            song["date"] = chart.date
            del song["image"]
            yield song
        print("Processed date ", chart.date)

if __name__ == "__main__":
    start_date = None

    if os.path.exists("datasets/billboard.csv"):
        df = pd.read_csv("datasets/billboard.csv")
        start_date = (datetime.strptime(df["date"].min(), "%Y-%m-%d") - timedelta(days=7)).date()
        print("Found existing billboard.csv, starting at date:", start_date)
    else:
        df = pd.DataFrame(data=[], columns=["title", "artist", "peakPos", "lastPos", "weeks", "rank", "isNew", "date"])
        df.to_csv("datasets/billboard.csv")
    
    try:
        for idx, song in enumerate(get_charts(start_date)):
            df = df.append(song, ignore_index=True)
    except Exception as e:
        print(e)
        pass
    
    print("Saving to billboard.csv")
    df.to_csv("datasets/billboard.csv")

