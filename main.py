import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.preprocessing import MinMaxScaler


def hit_non_hit(score):
    return 1
    # if score <= 10: # e.g. top 10
    #     return 1
    # else:
    #     return None


if __name__ == "__main__":
    seed = 42

    df = pd.read_csv("datasets/spotify_billboard_features.csv")

    non_hits = pd.read_csv("datasets/complete_project_data.csv")
    non_hits.columns = [c.lower() for c in non_hits.columns]
    non_hits = non_hits[non_hits["label"] == 0]

    ## decide when a song is a hit (can also use a non-hit dataset)
    for idx, row in df.iterrows():
        label = hit_non_hit(row.billboard_peakPos)
        df.at[idx, "label"] = label

    df = df.dropna(axis=0, subset=["label"])

    # sub-sample from hits to match non-hits
    df = df.sample(n=len(non_hits), random_state=seed, replace=False)

    print("No. of HITS:", len(df))
    print("No. of NON-HITS:", len(non_hits))

    features = [
        "danceability",
        "energy",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        # "time_signature",
    ]

    ## scale features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    non_hits[features] = scaler.fit_transform(non_hits[features])

    X_non = non_hits[features].values
    y_non = non_hits["label"].values.reshape(-1, 1)

    X = df[features].values
    y = df["label"].values.reshape(-1, 1)

    X = np.vstack([X, X_non])
    y = np.vstack([y, y_non])

    print("Label distribution:")
    print(np.unique(y, return_counts=True))

    ## train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )

    ## classifiers
    # clf = LogisticRegressionCV(cv=5, random_state=seed, max_iter=200)
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    # clf = RandomForestClassifier(random_state=seed, max_depth=2)
    clf.fit(X_train, y_train)

    ## scores
    test_acc = clf.score(X_test, y_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    print("(Test) Accuracy", test_acc)
    print("(Test) Precision", precision)
    print("(Test) Recall", recall)
    print("(Test) Average precision (AP)", average_precision)
    print("(Test) ROC-AUC", roc_auc)
