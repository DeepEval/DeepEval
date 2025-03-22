import keras
from keras.layers import Input, Conv2D, Add, Dense, Flatten, Concatenate, BatchNormalization

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Three sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Path 2: Directly processing the input with a convolutional layer
    conv_shortcut = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding the outputs of both paths
    added_output = Add()([conv3, conv_shortcut])

    # Batch normalization and flatten the result
    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model