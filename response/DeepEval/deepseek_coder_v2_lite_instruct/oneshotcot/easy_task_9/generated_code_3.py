import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Add 3x3 depthwise separable convolutional layer for feature extraction
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(conv1x1_1)

    # Step 4: Add another 1x1 convolutional layer to reduce dimensionality
    conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)

    # Step 5: Add the processed output from conv1x1_2 to the original input layer
    added = Add()([input_layer, conv1x1_2])

    # Step 6: Flatten the result
    flatten_layer = Flatten()(added)

    # Step 7: Add a fully connected layer to generate the final classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 8: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model