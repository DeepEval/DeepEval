import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block_1(input_tensor):
        weights = Dense(units=128, activation='relu')(GlobalAveragePooling2D()(input_tensor))
        weights = Dense(units=128, activation='relu')(weights)
        weights = Reshape(target_shape=(1, 1, 128))(weights)
        weighted_output = Multiply()([input_tensor, weights])
        return weighted_output
    
    # Block 2
    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
        main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        branch = block_1(input_tensor)
        adding_layer = Add()([main_path, branch])
        return adding_layer
    
    block2_output = block_2(input_tensor=input_layer)
    
    # Flatten and classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model