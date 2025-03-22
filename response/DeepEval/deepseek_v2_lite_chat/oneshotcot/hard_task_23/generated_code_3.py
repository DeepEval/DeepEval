import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def branch1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1
    
    branch1_output = branch1(conv1)

    def branch2(input_tensor):
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool)
        transpose = keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(conv2)
        return Concatenate()([transpose, conv2])
    
    branch2_output = branch2(input_tensor=branch1_output)

    def branch3(input_tensor):
        pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool)
        transpose = keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(conv3)
        return Concatenate()([transpose, conv3])
    
    branch3_output = branch3(input_tensor=branch1_output)

    concat = Concatenate()([branch1_output, branch2_output, branch3_output])
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    batch_norm = BatchNormalization()(conv4)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model