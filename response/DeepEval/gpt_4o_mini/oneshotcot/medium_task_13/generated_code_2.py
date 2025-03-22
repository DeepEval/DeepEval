import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second convolutional layer concatenated with the output of the first layer
    conv2_input = Concatenate()([input_layer, conv1])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_input)

    # Third convolutional layer concatenated with the output of the second layer
    conv3_input = Concatenate()([conv2_input, conv2])
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_input)

    # Flatten the output from the last convolutional layer
    flatten_layer = Flatten()(conv3)

    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    # Output layer with softmax activation for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()