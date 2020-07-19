import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def hit_non_hit(score):
    return 1
    # if score <= 10: # e.g. top 10
    #     return 1
    # else:
    #     return None


def feature_importance(clf, features):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print(
            "%d. feature %s (%d) (%f)"
            % (f + 1, features[f], indices[f], importances[indices[f]])
        )

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(
        range(X.shape[1]),
        importances[indices],
        color="r",
        yerr=std[indices],
        align="center",
    )
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, help="Random seed")
    parser.add_argument(
        "--hits",
        default="./datasets/spotify_billboard_features.csv",
        help="CSV file containing features of hit songs",
    )
    parser.add_argument(
        "--nonhits",
        default="./datasets/data_sample.csv",
        help="CSV file containing features of non-hit songs",
    )

    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "neural_network"],
        help="Classifier to use",
    )
    parser.add_argument(
        "--holdout_year", help="Which year's hit songs to withold for testing"
    )
    args = parser.parse_args()

    ## load hit/non-hit datasets
    hits = pd.read_csv(args.hits)
    hits["billboard_date"] = pd.to_datetime(hits["billboard_date"])
    hits["label"] = 1

    if args.holdout_year:
        hits_holdout = hits[hits["billboard_date"].dt.year == args.holdout_year]
        hits = hits[hits["billboard_date"].dt.year != args.holdout_year]

    non_hits = pd.read_csv(args.nonhits)
    non_hits.columns = [c.lower() for c in non_hits.columns]
    non_hits["label"] = 0

    ## sub-sample from hits to match non-hits
    ## otherwise, sub-sample from non-hits to match hits
    if len(hits) > len(non_hits):
        hits = hits.sample(n=len(non_hits), random_state=args.seed, replace=False)
    elif len(non_hits) > len(hits):
        non_hits = non_hits.sample(n=len(hits), random_state=args.seed, replace=False)

    print("No. of HITS:", len(hits))
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

    ## concatenate hits and non-hits dataframes
    all_df = pd.concat([hits, non_hits])

    ## scale features
    # scaler = MinMaxScaler()
    # all_df[features] = scaler.fit_transform(all_df[features])

    X = all_df[features].values
    y = all_df["label"].values

    ## train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, shuffle=True
    )

    ## classifiers
    if args.classifier == "logistic_regression":
        clf = LogisticRegression(random_state=args.seed, max_iter=1000)
    elif args.classifier == "random_forest":
        clf = RandomForestClassifier(random_state=args.seed, max_depth=2)
    elif args.classifier == "neural_network":
        clf = MLPClassifier(
            random_state=args.seed,
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(100,),
        )

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

    if args.classifier == "random_forest":
        feature_importance(clf, features)

    if args.holdout_year:
        print("Testing for hold-out hit songs of year:", args.holdout_year)

        X_holdout = hits_holdout[features].values
        y_holdout = hits_holdout["label"].values

        test_acc = clf.score(X_holdout, y_holdout)
        y_score = clf.predict_proba(X_holdout)[:, 1]
        y_pred = clf.predict(X_holdout)

        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        average_precision = average_precision_score(y_holdout, y_score)
        # roc_auc = roc_auc_score(y_holdout, y_score)

        print("(Test) Accuracy", test_acc)
        print("(Test) Precision", precision)
        print("(Test) Recall", recall)
        print("(Test) Average precision (AP)", average_precision)

        # print("(Test) ROC-AUC", roc_auc)
