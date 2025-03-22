import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)

    # First parallel block
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    batch_norm2 = BatchNormalization()(block1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)

    # Second parallel block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    batch_norm3 = BatchNormalization()(block2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch_norm3)

    # Third parallel block
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    batch_norm4 = BatchNormalization()(block3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(batch_norm4)

    # Concatenate the outputs of the parallel blocks
    concat = Concatenate()(list(block1))
    concat = Concatenate()(list(block2))
    concat = Concatenate()(list(block3))
    concat_pooled = Concatenate()([concat, pool1, pool2, pool3])

    # Batch normalization and flattening
    batch_norm_concat = BatchNormalization()(concat_pooled)
    flatten = Flatten()(batch_norm_concat)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()