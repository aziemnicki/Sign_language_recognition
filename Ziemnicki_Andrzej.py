import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
from joblib import load
from sklearn.preprocessing import LabelEncoder
import os

# mlflow.autolog()
def main():
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
    data.dropna( inplace=True)
    Y = data['letter']
    X = data.drop('letter', axis=1)
    X = X.iloc[:, 1:]
    results_list = []
    directory = os.path.join(os.getcwd(), 'training/model')
    results = list()
    model_path = os.path.join(directory, "klasyfikator.pkl")
    model = load(model_path)
    results.append(model.predict(X))
    results_list = [result.tolist() for result in results]

    y_pred = model.predict(X)
    f1 = f1_score(Y, y_pred, average='macro')
    results_list.append({'F1 score': f1})


    with results_file.open('w') as output_file:
        json.dump(results_list, output_file, indent=4)


if __name__ == '__main__':
    main()
