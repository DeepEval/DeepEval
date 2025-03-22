import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Input layer
    input_img = Input(shape=input_shape)

    # Branch path: 3x3 convolution and global average pooling
    branch1 = Conv2D(64, (3, 3), activation='relu')(input_img)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    branch1 = GlobalAveragePooling2D()(branch1)

    # Main path: Global average pooling
    branch2 = GlobalAveragePooling2D()(input_img)

    # Combine the outputs of both paths
    combined = keras.layers.concatenate([branch1, branch2])

    # Add fully connected layers
    dense1 = Dense(512, activation='relu')(combined)
    dense2 = Dense(256, activation='relu')(dense1)
    dense3 = Dense(128, activation='relu')(dense2)

    # Output layer
    output = Dense(10, activation='softmax')(dense3)

    # Define the model
    model = Model(inputs=input_img, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and return the constructed model
model = dl_model()