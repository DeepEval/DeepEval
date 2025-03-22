import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     
    # step 1: add input layer
    input_layer = Input(shape=(28, 28, 1))

    # step 2: define the block
    def block(input_tensor):
        # step 2.1: depthwise separable convolution
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # step 2.2: 1x1 convolution
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        
        # step 2.3: dropout layer
        conv = Dropout(0.2)(conv)
        
        return conv
    
    # step 3: add three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # step 4: concatenate the outputs of the branches
    output_tensor = Concatenate()([branch1, branch2, branch3])

    # step 5: add batch normalization
    batch_norm = BatchNormalization()(output_tensor)

    # step 6: flatten the output
    flatten_layer = Flatten()(batch_norm)

    # step 7: add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # step 8: add dropout layer
    dense1 = Dropout(0.2)(dense1)

    # step 9: add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # step 10: build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model