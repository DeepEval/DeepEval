from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, Reshape
from tensorflow.keras import Model

def dl_model():
    inputs = Input(shape=(28, 28, 1))

    # Block 1
    block1_conv1 = Conv2D(20, (1, 1), padding='same', activation='relu')(inputs)
    block1_pool1 = MaxPooling2D((1, 1), strides=(1, 1))(block1_conv1)
    block1_conv2 = Conv2D(20, (2, 2), padding='same', activation='relu')(block1_pool1)
    block1_pool2 = MaxPooling2D((2, 2), strides=(2, 2))(block1_conv2)
    block1_conv3 = Conv2D(20, (4, 4), padding='same', activation='relu')(block1_pool2)
    block1_pool3 = MaxPooling2D((4, 4), strides=(4, 4))(block1_conv3)
    block1_outputs = concatenate([block1_pool1, block1_pool2, block1_pool3])

    # Block 2
    block2_conv1 = Conv2D(40, (1, 1), padding='same', activation='relu')(block1_outputs)
    block2_conv2 = Conv2D(40, (3, 3), padding='same', activation='relu')(block2_conv1)
    block2_conv3 = Conv2D(40, (5, 5), padding='same', activation='relu')(block2_conv2)
    block2_pool = MaxPooling2D((3, 3), strides=(3, 3))(block2_conv3)
    block2_outputs = concatenate([block2_conv1, block2_conv2, block2_conv3, block2_pool])

    # Block 3
    block3_flatten = Flatten()(block2_outputs)
    block3_dense = Dense(128, activation='relu')(block3_flatten)
    block3_reshape = Reshape((1, 1, 128))(block3_dense)

    # Block 4
    block4_conv1 = Conv2D(40, (1, 1), padding='same', activation='relu')(block3_reshape)
    block4_conv2 = Conv2D(40, (3, 3), padding='same', activation='relu')(block4_conv1)

    # Output
    outputs = Flatten()(block4_conv2)
    outputs = Dense(10, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model