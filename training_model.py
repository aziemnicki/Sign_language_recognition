import time
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import mlflow
from sklearn.metrics import f1_score
from joblib import dump


# mlflow.autolog()
def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    results_file = Path(args.results_file)

    raw_data = pd.read_csv(dataset_path)
    # print(raw_data.info())
    # print(raw_data.describe())
    # print(raw_data.head(5))
    data = raw_data.copy()
    data.rename(columns={'Unnamed: 0': 'Number'}, inplace=True)
    data.rename(columns={'handedness.label': 'Hand'}, inplace=True)
    encoder = LabelEncoder()
    data['Hand'] = encoder.fit_transform(data['Hand'])
    # print(data['Hand'])
    Y = data['letter']
    X = data.drop('letter', axis=1)
    # X = X.drop('number', axis=1)
    X = X.iloc[:, 1:]
    print(X)
    # print(f' Y {Y}')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)

    results = dict()
    classif = MLPClassifier(activation='logistic',hidden_layer_sizes=200, max_iter=30000)
    class_sigmoid = CalibratedClassifierCV(classif,  method="sigmoid")

    clfs = [
        # SVC,
        # DecisionTreeClassifier,
        # RandomForestClassifier,
        # GradientBoostingClassifier,
        MLPClassifier
    ]

    # params = {'activation': ['logistic', 'relu'],  'hidden_layer_sizes': [100, 200, 300, 500],
    #           'max_iter': [2000, 5000, 10000, 25000, 50000],'n_iter_no_change': [5, 10, 15, 20]
    # }
    # klas = GridSearchCV(MLPClassifier(), params, cv=10)
    # klas.fit(X_train, y_train)
    # sorted_results = sorted(zip(klas.cv_results_['mean_test_score'], klas.cv_results_['params']), reverse=True)
    # for mean_score, params in sorted_results:
    #     print(f"Mean score: {mean_score:.4f}")
    #     print(f"Params: {params}")
    #     print("----------------------------------------")
    #
    for clf in clfs:
        mdl = Pipeline([
            ('min_max_scaler', MinMaxScaler()),
            ('standard_scaler', StandardScaler()),
            ('classifier', class_sigmoid)
        ])
        mdl.fit(X_train, y_train)
        results[clf.__name__] = mdl.score(X_test, y_test)
        y_pred = mdl.predict(X_test)

    print(results)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)

    dump(mdl, 'klasyfikator.pkl')


    stop = time.time()
    print(f'Elapsed time: {stop-start} seconds')
if __name__ == '__main__':
    main()
