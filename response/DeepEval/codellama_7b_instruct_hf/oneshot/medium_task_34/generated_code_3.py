import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First part: feature extraction and classification
    # Convolutional layer with max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Convolutional layer with dropout and max pooling
    conv2 = Conv2D(32, (3, 3), activation='relu')(pool1)
    drop1 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2))(drop1)

    # Convolutional layer with transposed convolution and skip connections
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
    transposed_conv1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
    skip1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(conv2)
    skip2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(conv1)
    skip3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input_layer)
    add1 = keras.layers.Add()([transposed_conv1, skip1])
    add2 = keras.layers.Add()([add1, skip2])
    add3 = keras.layers.Add()([add2, skip3])

    # Flatten and dense layers for classification
    flat = Flatten()(add3)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model