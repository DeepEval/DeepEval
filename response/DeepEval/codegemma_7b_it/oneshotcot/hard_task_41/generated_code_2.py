import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dense, Dropout, Reshape
from keras.regularizers import l2

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    # Regularize each pooling result using dropout
    path1 = Dropout(0.25)(path1)
    path2 = Dropout(0.25)(path2)
    path3 = Dropout(0.25)(path3)

    # Flatten and concatenate pooling outputs
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)
    concat = Concatenate()([path1, path2, path3])

    # Block 2
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 3x3 convolution
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(branch4)

    # Concatenate and fuse branch outputs
    concat_branch = Concatenate()([branch1, branch2, branch3, branch4])
    concat_branch = Flatten()(concat_branch)

    # Reshape output from block 1 for compatibility with block 2
    block1_output = Reshape((7, 7, 32))(concat)

    # Concatenate outputs from block 1 and block 2
    concat_final = Concatenate()([block1_output, concat_branch])

    # Fully connected layer
    dense = Dense(units=128, activation='relu', kernel_regularizer=l2(0.001))(concat_final)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model