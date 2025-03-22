import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers import Reshape, Multiply, Add
from keras import regularizers

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):

        pool = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(input_tensor)
        pool = Reshape(target_shape=(1, 1, 3))(pool)
        weights1 = Dense(units=10, activation='relu')(pool)
        weights1 = Dense(units=3, activation='relu')(weights1)
        weights1 = Reshape(target_shape=(3, 1, 1))(weights1)
        weighted_output = Multiply()([input_tensor, weights1])
        weighted_output = Reshape(target_shape=(32, 32, 3))(weighted_output)

        return weighted_output
    
    def block2(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

        return max_pooling
    
    weighted_output = block1(input_layer)
    block2_output = block2(weighted_output)

    # Connect a branch from Block 1 to Block 2
    branch = block1(input_layer)

    # Fuse the outputs from the main path and the branch
    combined_output = Add()([block2_output, branch])

    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model