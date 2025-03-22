import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution and 3x3 convolution
    branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1_conv1)
    branch1_dropout = Dropout(0.5)(branch1_conv2)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_conv4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2_conv3)
    branch2_dropout = Dropout(0.5)(branch2_conv4)

    # Branch 3: Max pooling
    branch3_maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch3_dropout = Dropout(0.5)(branch3_maxpool)

    # Concatenate all branches
    concatenated = Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model