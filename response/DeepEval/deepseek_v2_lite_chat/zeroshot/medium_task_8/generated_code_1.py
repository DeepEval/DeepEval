import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Input layer
input_layer = Input(shape=(32, 32, 3))

# Main path
split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
extracted = split1[0] * 0.25 + split1[1] * 0.25 + split1[2] * 0.5  # Adjust these weights according to your requirements
extracted = Conv2D(32, 3, activation='relu')(extracted)
extracted = MaxPooling2D()(extracted)

combined = concatenate([split1[0], extracted, split1[2]])
combined = Conv2D(64, 3, activation='relu')(combined)
combined = MaxPooling2D()(combined)

# Branch path
branch_extracted = split1[1]  # Use the second group from the split
branch_combined = concatenate([branch_extracted, combined])
branch_combined = Conv2D(64, 3, activation='relu')(branch_combined)

# Fusion of main and branch paths
output = concatenate([combined, branch_combined])
output = Flatten()(output)
output = Dense(10, activation='softmax')(output)  # Assuming 10 classes

# Model construction
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()

return model

# Instantiate and return the model
model = dl_model()