import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))

    # Define a main path with three parallel paths
    def main_path(input_tensor):
        split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        path1 = layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        path2 = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        path3 = layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        output_tensor = layers.Concatenate()([path1, path2, path3])
        return output_tensor
    
    # Define a branch path
    branch_path = layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Get the outputs from the main and branch paths
    main_output = main_path(input_layer)
    branch_output = branch_path
    
    # Add the outputs from both paths
    add_output = layers.Add()([main_output, branch_output])
    
    # Flatten the output
    flatten_layer = layers.Flatten()(add_output)
    
    # Add dense layers for classification
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model