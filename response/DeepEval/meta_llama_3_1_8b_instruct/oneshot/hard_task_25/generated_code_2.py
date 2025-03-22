import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Add, Dense, Conv2DTranspose, BatchNormalization, Activation

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Split into three branches
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)

    # Concatenate outputs of all branches
    output_main = Concatenate()([branch1, branch2, branch3])

    # Apply 1x1 convolutional layer to form main path output
    output_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(output_main)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main path and branch path outputs
    output = Add()([output_main, branch_path])

    # Flatten output
    output = Flatten()(output)

    # Apply fully connected layer for 10-class classification
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model