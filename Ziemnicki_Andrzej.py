import numpy as np
import time
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
###############################################
from sklearn.neural_network import MLPClassifier


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    results_file = Path(args.results_file)

    raw_data = pd.read_csv(dataset_path)
    print(raw_data.info())
    #print(raw_data.describe())
    print(raw_data.head(5))
    #TODO 'letter;
    data = raw_data.copy()
    data.rename(columns={'Unnamed: 0': 'Number'}, inplace=True)
    data.rename(columns={'handedness.label': 'Hand'}, inplace=True)
    encoder = LabelEncoder()
    #mapping = {'Right': 0.0, 'Left': 1.0}
    data['Hand'] = encoder.fit_transform(data['Hand'])
    print(data['Hand'])
    Y = data['letter']
    X = data.drop('letter', axis=1)
    #print(f' Y {Y}')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)


    results = dict()

    classif = MLPClassifier(activation='logistic',hidden_layer_sizes=200, max_iter=5000 )
    linear_svc = LinearSVC(max_iter=100000)
    clfs = [
        #LinearSVC,
        # SVC,
        # RandomForestClassifier,
        # DecisionTreeClassifier
        MLPClassifier
    ]


    for clf in clfs:
        mdl = Pipeline([
            ('min_max_scaler', MinMaxScaler()),
            ('standard_scaler', StandardScaler()),
            ('classifier', classif)  # RandomForestClassifier(), LinearSVC(), DecisionTreeClassifier()
        ])
        mdl.fit(X_train, y_train)
        # X_train = mdl.transform(X_train,y_train) #nie dziala juz jak jest classifier w pipelinie
        results[clf.__name__] = mdl.score(X_test, y_test)

    print(results)




    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)

    stop = time.time()
    print(f'Elapsed time: {stop-start} seconds')
if __name__ == '__main__':
    main()
