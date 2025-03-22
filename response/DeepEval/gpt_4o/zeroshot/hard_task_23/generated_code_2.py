from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 Convolutional Layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # First Branch: Local Feature Extraction with two 3x3 convolutions
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Second Branch: Downsample, 3x3 Conv, then Upsample
    branch2 = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(branch2)

    # Third Branch: Same as Second Branch
    branch3 = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 Convolutional Layer for feature refinement
    refined = Conv2D(128, (1, 1), activation='relu')(concatenated)

    # Flatten and Fully Connected Layer for classification
    flattened = Flatten()(refined)
    output = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Load CIFAR-10 data for demonstration purposes
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()