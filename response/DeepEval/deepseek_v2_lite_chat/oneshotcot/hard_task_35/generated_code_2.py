import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1 - Similar block as described
    def block1(input_tensor):
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers
        fc1 = Dense(units=128, activation='relu')(avg_pool)
        fc2 = Dense(units=64, activation='relu')(fc1)
        return fc2  # weights output from these layers will match input's shape
    
    # Branch 2 - Similar block as described
    def block2(input_tensor):
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers
        fc1 = Dense(units=64, activation='relu')(avg_pool)
        fc2 = Dense(units=32, activation='relu')(fc1)
        return fc2  # weights output from these layers will match input's shape
    
    # Define the block
    def block(input_tensor):
        weights_branch1 = block1(input_tensor)
        weights_branch2 = block2(input_tensor)
        # Concatenate the outputs
        concat = Concatenate()([weights_branch1, weights_branch2])
        # Reshape and multiply with the input
        reshaped_concat = Flatten()(concat)
        multiplied_concat = Dense(units=len(input_tensor.shape[1:]),
                                   kernel_initializer='zeros',
                                   kernel_regularizer=keras.regularizers.get(
                                       'adam'),
                                   bias_regularizer=keras.regularizers.get('adam'))(reshaped_concat)
        return multiplied_concat
    
    # Model construction
    input_layer = input_layer.reshape([input_layer.shape[0], -1])  # Flatten input
    output = block(input_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model