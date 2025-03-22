import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, GlobalMaxPooling2D, Multiply, Add, Flatten
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block_1(input_tensor):
        # Split input into three groups along the last dimension
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(split_layer[0])
        path1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path1)
        path1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(path1)
        
        path2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(split_layer[1])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(path2)
        
        path3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(split_layer[2])
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(path3)

        concatenated = Concatenate()([path1, path2, path3])
        return concatenated

    block_1_output = block_1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(block_1_output)

    # Block 2
    def block_2(input_tensor, original_input):
        # Global Max Pooling
        pooled = GlobalMaxPooling2D()(input_tensor)
        
        # Fully connected layers to generate channel-matching weights
        dense1 = Dense(units=16, activation='relu')(pooled)
        dense2 = Dense(units=3, activation='sigmoid')(dense1)
        
        # Reshape weights to match the shape of adjusted output
        reshaped_weights = Lambda(lambda x: tf.reshape(x, (-1, 1, 1, 3)))(dense2)
        
        # Multiply weights with the adjusted output
        weighted_output = Multiply()([input_tensor, reshaped_weights])
        
        # Addition with the direct branch from input
        branch_output = Add()([weighted_output, original_input])
        
        return branch_output

    block_2_output = block_2(transition_conv, input_layer)

    # Final Fully Connected Layer for Classification
    flatten_layer = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model