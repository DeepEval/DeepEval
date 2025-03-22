from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the main path of the model
inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (1, 1), padding="same")(inputs)

# Branch 1: 3x3 convolutional layer
branch1 = Conv2D(32, (3, 3), padding="same")(x)

# Branch 2: Max Pooling
branch2 = MaxPooling2D()(x)

# Branch 3: Max Pooling + Convolution + Upsampling
branch3 = MaxPooling2D()(branch2)
branch3 = Conv2D(32, (3, 3), padding="same")(branch3)
branch3 = UpSampling2D()(branch3)

# Concatenate the outputs from all branches
x = Concatenate()([branch1, branch2, branch3])

# Final 1x1 convolutional layer
x = Conv2D(64, (1, 1), padding="same")(x)

# Branch path
branch_path_input = Input(shape=(32, 32, 3))
branch_x = Conv2D(64, (1, 1), padding="same")(branch_path_input)

# Add the branch path outputs to the main path
x = Concatenate()([x, branch_x])

# Flatten and pass through two fully connected layers
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Construct the model
model = Model(inputs=[inputs, branch_path_input], outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

return model