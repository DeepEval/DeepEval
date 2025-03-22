import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path blocks
    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(input_tensor, 3, axis=1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[0])
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv1)
        output1 = tf.concat([conv1, split1[1], split1[2]], axis=-1)
        return output1
    
    def block2(input_tensor):
        shape = K.int_shape(input_tensor)
        output_shape = (shape[1], shape[2], 3)
        reshape = Lambda(lambda x: K.reshape(x, output_shape))
        permute = Permute((3, 1, 2))(reshape(input_tensor))
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(permute)
        depthwise = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(conv2)
        return depthwise
    
    def block3(input_tensor):
        conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(input_tensor)
        return conv3
    
    # Branch path
    branch = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Main path
    block_output1 = block1(input_layer)
    block_output2 = block2(block_output1)
    block_output3 = block3(block_output2)
    main_output = Concatenate()([block_output1, block_output2, block_output3])
    
    # Combine outputs
    combined_output = Concatenate()([main_output, branch])
    
    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(combined_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()