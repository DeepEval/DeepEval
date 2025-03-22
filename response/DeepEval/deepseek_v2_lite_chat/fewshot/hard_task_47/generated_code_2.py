import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block: Split into three groups and apply depthwise separable convolutions
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_tensor)
        concat = Concatenate()([conv1, conv2, conv3, pool1, pool2])
        return concat

    # Second block: Multiple branches for feature extraction
    def branch(input_tensor, depth, kernel_size):
        conv = Conv2D(depth, kernel_size, padding='same', activation='relu')(input_tensor)
        return conv

    # Branches with different configurations
    branch1 = branch(input_tensor=block1_output, depth=64, kernel_size=(1, 1))
    branch2 = branch(input_tensor=block1_output, depth=64, kernel_size=(3, 3))
    branch3 = branch(input_tensor=block1_output, depth=64, kernel_size=(5, 5))
    branch4 = branch(input_tensor=block1_output, depth=64, kernel_size=(1, 7))
    branch5 = branch(input_tensor=block1_output, depth=64, kernel_size=(7, 1))
    branch6 = branch(input_tensor=block1_output, depth=64, kernel_size=(3, 3))

    # Concatenate all branches
    concat_output = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6])

    # Flatten and output layer
    flatten = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])