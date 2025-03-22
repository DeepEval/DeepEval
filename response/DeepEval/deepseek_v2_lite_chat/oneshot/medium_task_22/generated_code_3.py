import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the dataset
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)

    # Branch 1: 1x1 convolution, followed by 3x3 convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv3)
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)

    # Branch 2: Max pooling
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Concatenate the features from all branches
    concat = Concatenate()([conv3, conv1_3, conv1_1, pool])

    # Batch normalization and flattening
    bn = BatchNormalization()(concat)
    flat = Flatten()(bn)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    outputs = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])