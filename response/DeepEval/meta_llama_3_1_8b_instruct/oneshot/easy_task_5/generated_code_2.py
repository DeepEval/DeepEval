import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Reduce the input dimensionality with a 1x1 convolution
    conv_reduce_dim = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract features using a 3x3 convolutional layer
    conv_extract_features = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_reduce_dim)

    # Restore the dimensionality of the feature map with another 1x1 convolution layer
    conv_restore_dim = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_extract_features)

    # Flatten the output
    flatten_layer = Flatten()(conv_restore_dim)

    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model