import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, AveragePooling2D, Dense, Flatten, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        return conv1, avg_pool1

    # Second block
    def block_2(input_tensor):
        avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        avg_pool2_flat = Flatten()(avg_pool2)
        dense1 = Dense(units=128, activation='relu')(avg_pool2_flat)
        reshape = Reshape((128,))(dense1)
        dense2 = Dense(units=64, activation='relu')(reshape)
        weights = Dense(units=64)(dense2)
        refined_input = keras.layers.multiply([input_tensor, keras.layers.expand_dims(weights)])
        return refined_input

    # First block processing
    conv1, avg_pool1 = block_1(input_tensor=input_layer)
    added_feature_map = Add()([conv1, avg_pool1])

    # Second block processing
    refined_input = block_2(input_tensor=added_feature_map)
    flat_refined_input = Flatten()(refined_input)
    dense_output = Dense(units=10, activation='softmax')(flat_refined_input)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model