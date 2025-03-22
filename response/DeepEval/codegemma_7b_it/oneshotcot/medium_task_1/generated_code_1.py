import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Convolutional layer 2
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Output features directly added with input layer
    output_features = Concatenate()([max_pooling, input_layer])

    # Batch normalization layer
    bath_norm = BatchNormalization()(output_features)

    # Flatten layer
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layer 1
    dense1 = Dense(units=64, activation='relu')(flatten_layer)

    # Fully connected layer 2
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model