import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolutional Layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # 3x1 Convolutional Layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # 1x3 Convolutional Layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Restore number of channels
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)

    # Dropout
    dropout = Dropout(0.2)(conv5)

    # Add processed features with original input
    add = Concatenate()([dropout, input_layer])

    # Flatten
    flatten = Flatten()(add)

    # Fully Connected Layers
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model