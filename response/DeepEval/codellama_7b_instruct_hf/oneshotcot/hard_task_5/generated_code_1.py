from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define main path
    main_path = input_layer

    # Block 1
    # Split input into three groups and process each group with a 1x1 convolution
    block1 = Lambda(lambda x: tf.split(x, 3, axis=3))(main_path)
    block1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)

    # Concatenate outputs from each group
    block1 = Concatenate()(block1)

    # Block 2
    # Reshape feature map and swap third and fourth dimensions
    block2 = Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3)))(block1)
    block2 = Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(block2)
    block2 = Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3)))(block2)

    # 3x3 depthwise separable convolution
    block2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)

    # Block 3
    # Add branch that connects directly to the input
    branch = input_layer

    # Concatenate outputs from main path and branch
    output = Concatenate()([main_path, branch])

    # Add fully connected layer
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    # Define model
    model = Model(inputs=input_layer, outputs=output)

    return model