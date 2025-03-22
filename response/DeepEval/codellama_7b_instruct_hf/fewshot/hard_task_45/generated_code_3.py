import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Lambda, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Depthwise separable convolutional layers with varying kernel sizes (1x1, 3x3, and 5x5)
    conv1_layer = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2_layer = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3_layer = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    # Concatenate the outputs from all three groups
    concat_layer = Concatenate()([conv1_layer, conv2_layer, conv3_layer])

    # Second block
    # Feature extraction branches
    branch1_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    branch2_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    branch3_layer = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concat_layer)

    # Concatenate the outputs from all branches
    concat_layer = Concatenate()([branch1_layer, branch2_layer, branch3_layer])

    # Flatten the output
    flatten_layer = Flatten()(concat_layer)

    # Fully connected layers
    dense1_layer = Dense(units=128, activation='relu')(flatten_layer)
    dense2_layer = Dense(units=64, activation='relu')(dense1_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model