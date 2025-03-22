import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Generate the sequence
sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Split the sequence into training and testing sets
train_size = int(len(sequence) * 0.8)
train_seq, test_seq = sequence[:train_size], sequence[train_size:]

# Create a pandas DataFrame from the sequence
df = pd.DataFrame({'sequence': sequence})

# Split the DataFrame into training and testing sets
train_df, test_df = df[:train_size], df[train_size:]

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(train_df['sequence'], train_df['sequence'])

# Use the model to predict the next term in the sequence
prediction = model.predict([test_seq[0]])[0]

# Print the predicted value
print(prediction)