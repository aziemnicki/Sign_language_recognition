import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import json
import pandas as pd
from sklearn.metrics import f1_score
from joblib import load

def perform_processing(
        data: pd.DataFrame
) -> pd.DataFrame:
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data
    data.rename(columns={'Unnamed: 0': 'Number'}, inplace=True)
    data.rename(columns={'handedness.label': 'Hand'}, inplace=True)
    encoder = LabelEncoder()
    data['Hand'] = encoder.fit_transform(data['Hand'])
    X = data.iloc[:, 1:]

    directory = os.path.join(os.getcwd(), 'processing/model')
    results = list()
    model_path = os.path.join(directory, "klasyfikator.pkl")
    model = load(model_path)
    results.append(model.predict(X))
    results = np.ravel(results)



    # for the simplest approach generate a random DataFrame with proper column names and size
    # random_results = np.random.choice(
    #     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    #      't', 'u', 'v', 'w', 'x', 'y'],
    #     data.shape[0]
    # )
    # print(f'{random_results=}')

    predicted_data = pd.DataFrame(
        results,
        columns=['letter']
    )

    print(f'{predicted_data=}')

    return predicted_data
