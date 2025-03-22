import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolution and Pooling Layers
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)

    # Flatten and Concatenate Outputs
    flat_1x1 = Flatten()(pool_1x1)
    flat_2x2 = Flatten()(pool_2x2)
    flat_4x4 = Flatten()(pool_4x4)
    concat = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully Connected Layers
    dense1 = Dense(units=256, activation='relu')(concat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model