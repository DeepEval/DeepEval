import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    flattened_paths = [Flatten()(path) for path in [path1, path2, path3]]
    block1_output = Concatenate()(flattened_paths)

    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_output = Reshape((1, 1, 128))(dense1)  

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1) 

    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2) 

    branch3 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshaped_output)

    block2_output = Concatenate()([branch1, branch2, branch3])

    # Final Layers
    flatten_block2 = Flatten()(block2_output)
    dense2 = Dense(units=64, activation='relu')(flatten_block2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model