import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic block
    def basic_block(input_tensor):
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main = BatchNormalization()(main)
        main = Add()([branch, main])
        return main

    # Apply two basic blocks
    x = basic_block(x)
    x = basic_block(x)

    # Branch extraction
    branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Feature fusion
    x = Add()([x, branch])

    # Average pooling
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    
    # Flatten and fully connected layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model