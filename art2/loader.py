import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_data_from_file(path_to_file: str, norm: bool = False):
    df = pd.read_csv(path_to_file)
    # df = df.sample(frac=1)

    # if df.columns == 3:
    #     return df['x', 'y'].to_numpy()

    points = df[df.columns[0:-1]].to_numpy()
    labels = df[df.columns[-1]].to_numpy()

    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(points)
        points = pd.DataFrame(x_scaled)
        points = points.to_numpy()

    return points, labels