import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Feature extraction block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Feature extraction block 2
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv6)

    # Feature flattening and fully connected layers
    flatten_layer = Flatten()(max_pooling2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model