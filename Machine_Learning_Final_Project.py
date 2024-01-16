import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

dataset = pd.read_csv("C:\\Users\\sirin\\Downloads\\archive\\mushrooms.csv")
dataset.info()