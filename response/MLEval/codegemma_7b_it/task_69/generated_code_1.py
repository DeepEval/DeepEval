from tensorflow.keras.models import load_model
from sklearn.externals.estimator import KerasClassifier

# Load the compiled Keras model
model = load_model('path/to/keras_model.h5')

# Create an estimator from the Keras model
estimator = KerasClassifier(model=model)

# Example input data
input_data = np.array([[1, 2, 3], [4, 5, 6]])

# Predict using the estimator
output = estimator.predict(input_data)

# Print the output
print(output)

# Return the output if needed
# return output