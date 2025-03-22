import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1
    branch1_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch1_output = Concatenate()([branch1_conv1x1, branch1_conv3x3])
    branch1_output = Dropout(rate=0.5)(branch1_output)

    # Branch 2
    branch2_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv1x7 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch2_conv1x1)
    branch2_conv7x1 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch2_conv1x7)
    branch2_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch2_output = Concatenate()([branch2_conv1x1, branch2_conv7x1, branch2_conv3x3])
    branch2_output = Dropout(rate=0.5)(branch2_output)

    # Branch 3
    branch3_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch3_output = Dropout(rate=0.5)(branch3_output)

    # Concatenate all branches
    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    batch_norm_output = BatchNormalization()(concatenated_output)
    
    flatten_layer = Flatten()(batch_norm_output)
    
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense2 = Dropout(rate=0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model