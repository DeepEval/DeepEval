import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def method():
    # Sample data (replace with your actual data)
    data = {
        'feature1': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15, 16, np.nan, 18, 19, 20],
        'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(exclude=['object']).columns

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create Logistic Regression model
    model = LogisticRegression()

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the pipeline using the non-missing observations
    pipeline.fit(X_train.dropna(), y_train)

    return pipeline

# Call the method for validation
output = method()
print(output)