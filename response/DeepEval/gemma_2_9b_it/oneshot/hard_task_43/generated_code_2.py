import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat_pool1 = Flatten()(avg_pool1)
    flat_pool2 = Flatten()(avg_pool2)
    flat_pool4 = Flatten()(avg_pool4)

    combined_output = Concatenate()([flat_pool1, flat_pool2, flat_pool4])
    dense1 = Dense(units=128, activation='relu')(combined_output)
    reshaped_output = Reshape((1,128))(dense1)

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(branch1)

    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(branch2)

    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(branch3)

    concatenated_output = Concatenate()([branch1, branch2, branch3])

    # Output Layers
    dense2 = Dense(units=64, activation='relu')(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model