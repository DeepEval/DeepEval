import keras
from keras.layers import Input, GlobalAveragePooling2D, Lambda, Dense, Reshape, Concatenate, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        # Global average pooling to compress the input features
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers to produce weights whose dimension is equal to the channel of input
        weights = Dense(32, activation='relu')(avg_pool)
        weights = Dense(32, activation='relu')(weights)
        # Reshape the weights to match the input's shape
        weights = Reshape(target_shape=(1, 1, 32))(weights)
        # Element-wise multiplication with the block's input
        elementwise_mul = Lambda(lambda x: x[0] * x[1])([input_tensor, weights])
        return elementwise_mul
    
    # Two branches, each incorporating a same block
    branch1 = same_block(input_layer)
    branch2 = same_block(input_layer)
    
    # Concatenate the outputs from both branches
    concat = Concatenate()([branch1, branch2])
    
    # Flattening layer
    flatten = Flatten()(concat)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model