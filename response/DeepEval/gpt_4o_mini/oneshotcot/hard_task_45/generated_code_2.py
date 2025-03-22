import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda, BatchNormalization
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block with depthwise separable convolutions
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Depthwise separable convolutions with varying kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])

    # Concatenate the outputs of the first block
    block1_output = Concatenate()([path1, path2, path3])

    # Second block with multiple branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch4)

    # Concatenate outputs from the second block
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Batch normalization
    batch_norm = BatchNormalization()(block2_output)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model