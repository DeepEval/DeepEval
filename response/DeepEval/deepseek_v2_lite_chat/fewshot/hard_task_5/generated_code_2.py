import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Add, Flatten, Dense, Reshape, Permute

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch connecting directly to the input
    branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')(branch)

    # Main path blocks
    def block_1(input_tensor):
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[2])
        fused = keras.layers.Concatenate()([conv1_1, conv1_2, conv1_3])
        return fused

    def block_2(input_tensor):
        shape = keras.layers.Lambda(lambda tensor: tf.shape(tensor))[0](input_tensor)
        reshape = keras.layers.Lambda(lambda tensor: tf.reshape(tensor, (tensor[0], tensor[1], 3, -1)))(shape)
        shuffle = Permute((2, 3, 1))(reshape)
        conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(shuffle)
        return conv2_1

    def block_3(input_tensor):
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        return depthwise

    def block_4(input_tensor):
        merged = Add()([block_3(block_2(block_1(input_layer)))[0], branch])
        flatten = Flatten()(merged)
        dense = Dense(units=512, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(dense)
        return output

    block1 = block_1(input_tensor=input_layer)
    block2 = block_2(input_tensor=block1)
    block3 = block_3(block2)
    model = block_4(input_tensor=block3)

    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])