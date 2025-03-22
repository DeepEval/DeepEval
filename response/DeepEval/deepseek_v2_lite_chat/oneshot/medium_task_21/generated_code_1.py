import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D
from keras.models import Model
from keras.layers import Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch3)

    # Branch 4: Average pooling
    branch4 = MaxPooling2D(pool_size=(1, 1))(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch4)

    # Dropout layer applied at each branch
    branch1 = Dropout(0.5)(branch1)
    branch2 = Dropout(0.5)(branch2)
    branch3 = Dropout(0.5)(branch3)
    branch4 = Dropout(0.5)(branch4)

    # Concatenate the outputs from all branches
    output_tensor = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Pass through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(output_tensor)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model