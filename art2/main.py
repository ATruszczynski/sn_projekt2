from art2 import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
from sklearn import preprocessing

from art2 import loader
from art2.art import Art2
from astar_art import ART2


def analyse_clustering(labels, labels2):
    labels2 = np.array(labels2)
    for i in np.unique(labels2):
        print(f'Class {i} - {len(labels2[labels2==i])}')
    randScore = sm.adjusted_rand_score(labels, labels2)
    print(randScore)