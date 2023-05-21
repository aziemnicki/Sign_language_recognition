import numpy as np
import argparse
import json
from pathlib import Path
import pandas as pd

#raw_data = pd.read_csv('WZUM dataset.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    results_file = Path(args.results_file)
    raw_data = pd.read_csv(dataset_path)






    results = {}

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
