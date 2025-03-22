import keras
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape
from keras.regularizers import Dropout

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(maxpool1)
    maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(maxpool2)
    maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(maxpool3)
    dropout1 = Dropout(rate=0.2)(flatten1)
    dropout2 = Dropout(rate=0.2)(flatten2)
    dropout3 = Dropout(rate=0.2)(flatten3)
    output_tensor = Concatenate()([dropout1, dropout2, dropout3])

    # Block 2
    dense = Dense(units=64, activation='relu')(output_tensor)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped)
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped)
    conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(reshaped)
    output_tensor = Concatenate()([conv1, conv2, conv3, conv4])

    # Final classification layer
    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model