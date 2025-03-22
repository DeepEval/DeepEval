import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense, Lambda

def dl_model():
    # Input layer for CIFAR-10 dataset (32x32x3 images)
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting the input into three groups along the channel
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Convolutional layers with varying kernel sizes for each split
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensors[0])
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])

    # Apply dropout to reduce overfitting
    drop = Dropout(0.5)(Concatenate()([conv_1x1, conv_3x3, conv_5x5]))

    # Block 2: Four branches
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(drop)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(drop)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by 5x5 Convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(drop)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(branch3)

    # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(drop))
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate outputs from all branches for feature fusion
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the result and add fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model