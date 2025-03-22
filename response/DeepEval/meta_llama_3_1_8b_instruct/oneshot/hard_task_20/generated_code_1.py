import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Main path: Split the input into three groups and process each group separately
    def main_path(input_tensor):
        split_tensor = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        conv1x1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(split_tensor[0])
        conv3x3 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(split_tensor[1])
        conv5x5 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(split_tensor[2])
        output_tensor = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
        return output_tensor
    
    # Branch path: Process the input with a 1x1 convolutional layer
    branch_path = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Combine the outputs of the main and branch paths
    fused_features = layers.Add()([main_path(input_layer), branch_path])
    
    # Flatten the fused features
    flattened_features = layers.Flatten()(fused_features)
    
    # Perform classification using two fully connected layers
    dense1 = layers.Dense(64, activation='relu')(flattened_features)
    output_layer = layers.Dense(10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model