import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path with convolutional layers followed by a max pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
    
    # Branch path with global average pooling and channel weighting
    gap = GlobalAveragePooling2D()(max_pooling)
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    channel_weights = Reshape((1, 1, 32))(dense2)
    scaled_input = Multiply()([max_pooling, channel_weights])
    
    # Add the outputs of both paths
    added = Add()([max_pooling, scaled_input])
    
    # Final fully connected layers for classification
    flatten_layer = Flatten()(added)
    fc1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model