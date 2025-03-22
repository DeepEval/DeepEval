# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def method():
    # Assume X and y are your data
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10]
    })

    y = pd.Series([0, 0, 1, 1, 0])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-Hot Encode the target variable
    encoder = OneHotEncoder()
    y_train_oh = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_train_oh = to_categorical(y_train_oh)

    # Create a pipeline for preprocessing the data
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', SimpleImputer(strategy='constant', fill_value='missing'), categorical_features)
        ]
    )

    # Fit the preprocessor to the training data
    preprocessor.fit(X_train)

    # Transform the training and testing data using the preprocessor
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Convert the transformed data into a DataFrame
    X_train_indices = pd.DataFrame(X_train_transformed.toarray())
    X_test_indices = pd.DataFrame(X_test_transformed.toarray())

    # Create the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_indices.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train_oh.shape[1], activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(X_train_indices, y_train_oh, epochs=50, batch_size=32, verbose=0)

    # Return the output (in this case, the model's accuracy)
    output = model.evaluate(X_train_indices, y_train_oh)[1]
    return output

# Call the method for validation
output = method()
print("Model's accuracy:", output)