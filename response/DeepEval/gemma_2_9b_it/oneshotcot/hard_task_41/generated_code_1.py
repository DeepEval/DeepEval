import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Reshape, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(path1)
    flat2 = Flatten()(path2)
    flat3 = Flatten()(path3)

    dropout1 = Dropout(0.2)(flat1)
    dropout2 = Dropout(0.2)(flat2)
    dropout3 = Dropout(0.2)(flat3)

    concatenated_block1 = Concatenate()([dropout1, dropout2, dropout3])

    # Fully connected and reshape
    dense1 = Dense(128, activation='relu')(concatenated_block1)
    reshape_layer = Reshape((4, 32))(dense1)

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(reshape_layer)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)

    concatenated_block2 = Concatenate()([branch1, branch2, branch3])

    # Output Layers
    dense2 = Dense(units=64, activation='relu')(concatenated_block2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model