# Import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target