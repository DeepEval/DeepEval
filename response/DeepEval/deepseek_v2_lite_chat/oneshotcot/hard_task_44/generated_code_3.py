import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Split the input into three groups along the channel axis
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split1[0] = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
    split1[1] = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split1[1])
    split1[2] = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split1[2])
    drop1 = Dropout(0.2)(split1[0])  # Apply dropout to reduce overfitting

    # Concatenate the outputs from the three groups
    concat1 = Concatenate(axis=-1)(split1)
    # Add batch normalization and flatten layer
    bn_concat1 = BatchNormalization()(concat1)
    flat1 = Flatten()(bn_concat1)
    dense1 = Dense(units=128, activation='relu')(flat1)

    # Block 2
    split2 = Lambda(lambda x: tf.split(x, 5, axis=-1))(input_layer)
    split2[0] = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split2[0])
    split2[1] = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split2[1])
    split2[2] = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split2[2])
    split2[3] = MaxPooling2D(pool_size=(3, 3), strides=1)(split2[3])  # Max pooling with kernel size of 3x3
    split2[4] = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split2[4])
    split2[5] = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split2[5])
    split2[6] = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split2[6])
    split2[7] = MaxPooling2D(pool_size=(3, 3), strides=1)(split2[7])  # Max pooling with kernel size of 3x3

    concat2 = Concatenate(axis=-1)(split2)
    bn_concat2 = BatchNormalization()(concat2)
    flat2 = Flatten()(bn_concat2)
    dense2 = Dense(units=128, activation='relu')(flat2)

    # Fully connected layers
    dense3 = Dense(units=64, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model