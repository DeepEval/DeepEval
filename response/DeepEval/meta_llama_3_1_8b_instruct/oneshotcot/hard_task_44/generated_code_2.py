import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     

    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input into three groups along the channel
        split = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Feature extraction through convolutional with varying kernel sizes
        path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        path2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        path3 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        
        # Concatenate the outputs from the three groups
        output_tensor = layers.Concatenate()([path1, path2, path3])
        
        # Apply dropout to reduce overfitting
        output_tensor = layers.Dropout(0.2)(output_tensor)
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Define the four branches
        path1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path5 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Concatenate the outputs from all branches
        output_tensor = layers.Concatenate()([path1, path2, path3, path4, path5])
        
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Apply batch normalization
    block2_output = layers.BatchNormalization()(block2_output)
    
    # Apply max pooling
    block2_output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2_output)
    
    # Flatten the output
    flatten_layer = layers.Flatten()(block2_output)
    
    # Output layer
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model