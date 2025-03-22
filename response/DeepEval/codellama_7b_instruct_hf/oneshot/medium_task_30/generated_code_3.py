import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from keras.models import Model

def dl_model():
    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the first average pooling layer
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')

    # Define the second average pooling layer
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    # Define the third average pooling layer
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), activation='relu')(conv2)

    # Define the pooling layers
    pool1_out = pool1(conv1)
    pool2_out = pool2(conv2)
    pool3_out = pool3(conv3)

    # Define the flattening layer
    flatten = Flatten()

    # Define the concatenation layer
    concat = Concatenate()([pool1_out, pool2_out, pool3_out])

    # Define the fully connected layers
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model