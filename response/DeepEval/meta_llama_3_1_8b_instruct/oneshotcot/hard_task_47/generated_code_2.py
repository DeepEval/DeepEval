import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Define a custom function to split the input along the last dimension
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)
    
    # Apply the custom function using Lambda layer
    split_output = Lambda(split_input)(input_layer)

    # Apply depthwise separable convolutional layers for feature extraction
    path1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_output[2])

    # Apply batch normalization
    path1 = BatchNormalization()(path1)
    path2 = BatchNormalization()(path2)
    path3 = BatchNormalization()(path3)

    # Concatenate the outputs from the three groups
    output_tensor = Concatenate()([path1, path2, path3])

    # Define another custom function to process the input through three distinct branches
    def branch_process(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        
        path4 = layers.GlobalAveragePooling2D()(input_tensor)
        path4 = Dense(units=64, activation='relu')(path4)
        
        return [path1, path2, path3, path4]
    
    # Apply the custom function using Lambda layer
    branch_output = Lambda(branch_process)(output_tensor)
    
    # Concatenate the outputs from all branches
    final_output = Concatenate()([branch_output[0], branch_output[1], branch_output[2], branch_output[3]])

    # Apply batch normalization
    final_output = BatchNormalization()(final_output)

    # Flatten the output
    flatten_layer = Flatten()(final_output)

    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model