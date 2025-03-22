import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_output = Concatenate()([branch1_conv1, branch1_conv3])
    branch1_output = Dropout(rate=0.25)(branch1_output)

    # Branch 2
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv3)
    branch2_output = Dropout(rate=0.25)(branch2_conv4)

    # Branch 3
    branch3_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch3_output = Dropout(rate=0.25)(branch3_output)

    # Concatenating all branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model