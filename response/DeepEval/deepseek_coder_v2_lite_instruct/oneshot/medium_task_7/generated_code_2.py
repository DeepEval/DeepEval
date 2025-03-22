import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    batch_norm2 = BatchNormalization()(conv2)

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm2)
    batch_norm3 = BatchNormalization()(conv3)

    # Add outputs of first two convolutional layers and the third convolutional layer
    added_output = Add()([batch_norm1, batch_norm2, batch_norm3])

    # Flatten the result
    flatten_layer = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model