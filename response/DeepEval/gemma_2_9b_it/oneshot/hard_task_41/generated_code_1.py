import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Reshape, Dense, Concatenate

def dl_model():  
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool1 = Flatten()(pool1)
    pool1 = Dropout(0.2)(pool1)

    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool2 = Flatten()(pool2)
    pool2 = Dropout(0.2)(pool2)

    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    pool3 = Flatten()(pool3)
    pool3 = Dropout(0.2)(pool3)

    block1_output = Concatenate()([pool1, pool2, pool3])

    # Fully connected layer and reshape for Block 2
    dense1 = Dense(128, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 32))(dense1) 

    # Block 2
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)

    pool_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(reshape_layer)
    pool_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool_1)

    branch_outputs = [conv1_1, conv1_2, conv2_1, conv2_2, conv2_3, pool_1]

    block2_output = Concatenate()(branch_outputs)
    
    # Flatten and final classification layers
    flatten_layer = Flatten()(block2_output)
    dense3 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model