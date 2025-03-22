import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Concatenate

def dl_model():     
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    x = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    x1 = Flatten()(x)
    x1 = Dropout(0.2)(x1)

    x = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    x2 = Flatten()(x)
    x2 = Dropout(0.2)(x2)

    x = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    x3 = Flatten()(x)
    x3 = Dropout(0.2)(x3)

    block1_output = Concatenate()([x1, x2, x3])

    # Fully connected and reshape
    dense1 = Dense(units=64, activation='relu')(block1_output)
    reshape_layer = Reshape((1, 64))(dense1)
    
    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    branch3 = AveragePooling2D(pool_size=(1, 1), strides=1)(reshape_layer)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3) 
    
    block2_output = Concatenate()([branch1, branch2, branch3])

    flatten_layer = Flatten()(block2_output)
    dense2 = Dense(units=32, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model