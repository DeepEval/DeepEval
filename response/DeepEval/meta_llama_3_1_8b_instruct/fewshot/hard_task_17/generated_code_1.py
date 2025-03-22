import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape
from keras import backend as K

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Global average pooling
        pooled = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layer
        dense1 = Dense(units=3, activation='relu')(pooled)
        # Fully connected layer
        dense2 = Dense(units=3, activation='relu')(dense1)
        # Reshape weights to match input shape
        weights = Reshape(target_shape=(3, 3, 3))(dense2)
        # Weighted feature output
        weighted = Multiply()([input_tensor, weights])
        return weighted

    def block_2(input_tensor):
        # Convolutional layer
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Max pooling layer
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        # Convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        return conv2

    block1_output = block_1(input_tensor=input_layer)
    # Direct connection from Block 1 to Block 2
    branch = Add()([block1_output, block2_output])
    # Main path output
    main_output = block_2(input_tensor=block1_output)
    # Fuse main path and branch output through addition
    fused_output = Add()([main_output, branch])
    # Flatten output
    flat_output = K.flatten(fused_output)
    # Classification output
    output_layer = Dense(units=10, activation='softmax')(flat_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model