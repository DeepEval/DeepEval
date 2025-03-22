import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Basic Block
    def basic_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        activation = Activation('relu')(bn)
        branch = input_tensor 
        add = Add()([activation, branch])
        return add

    # First Basic Block
    x = basic_block(x)
    
    # Second Basic Block
    x = basic_block(x)

    # Branch
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch = basic_block(branch)
    
    # Feature Fusion
    x = Add()([x, branch])

    # Average Pooling
    x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    # Flatten and Fully Connected Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model