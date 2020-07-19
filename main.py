import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    seed = 42

    df = pd.read_csv("complete_project_data.csv")
    features = [
        "ArtistScore",
        "Danceability",
        "Energy",
        "Loudness",
        "Mode",
        # "Speechiness",
        "Acousticness",
        # "Instrumentalness",
        # "Liveness",
        # "Valence",
        "Tempo",
    ]

    ## replace categorial variable by one-hot encoding
    categoricals = [
        "Year",
        # "Month",
        # "Key"
    ]
    df = pd.get_dummies(df, columns=categoricals)
    cat_names = [k for c in categoricals for k in df.columns if c in k]
    features += cat_names

    ## standardize features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    ## get dataframe into array
    X = df[features].values
    y = df["Label"].values

    ## train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    ## classifiers
    # clf = LogisticRegressionCV(cv=5, random_state=seed, max_iter=200)
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)


    ## scores
    test_acc = clf.score(X_test, y_test)
    y_score = clf.decision_function(X_test)
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_score)

    print("(Test) Accuracy", test_acc)
    print("(Test) Precision", precision)
    print("(Test) Recall", recall)
    print("(Test) Average precision (AP)", average_precision)

