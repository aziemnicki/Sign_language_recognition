import numpy as np
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    results_file = Path(args.results_file)
    raw_data = pd.read_csv(dataset_path)
    print(raw_data.info())
    print(raw_data.describe())
    #TODO 'letter;
    data = raw_data.copy()
    Y = data['letter']
    X = data.drop('letter', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)




    results = {}

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
