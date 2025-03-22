import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Dense, Flatten, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group through convolutions
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(groups[0])
        conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_1)
        conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_3)
        
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(groups[1])
        conv3_4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_3)
        conv1_4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_4)
        
        conv1_5 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(groups[2])
        conv3_5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_5)
        conv1_6 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_5)
        
        # Concatenate the outputs
        concatenated = Concatenate(axis=-1)([conv1_2, conv1_4, conv1_6])
        return concatenated

    def transition_conv(input_tensor):
        # Adjust the number of channels to match the input layer
        conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        pooled = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return pooled

    def block_2(input_tensor):
        # Global max pooling
        pooled = MaxPooling2D(pool_size=(8, 8), padding='same')(input_tensor)
        
        # Generate weights through fully connected layers
        fc1 = Dense(units=64, activation='relu')(pooled)
        fc2 = Dense(units=32, activation='relu')(fc1)
        weights = Reshape((8, 8, 32))(fc2)
        
        # Multiply weights with the input to produce the main path output
        main_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        main_path_weighted = Multiply()([weights, main_path])
        
        # Branch directly from the input
        branch = input_tensor
        
        # Add the outputs from both paths
        added = Add()([main_path_weighted, branch])
        
        # Flatten and pass through fully connected layer for classification
        flattened = Flatten()(added)
        output_layer = Dense(units=10, activation='softmax')(flattened)
        
        return output_layer

    # Apply Block 1, Transition Convolution, and Block 2 sequentially
    block1_output = block_1(input_layer)
    transitioned = transition_conv(block1_output)
    final_output = block_2(transitioned)

    model = keras.Model(inputs=input_layer, outputs=final_output)
    return model