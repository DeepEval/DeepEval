import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # First block
    def split_convolution(input_tensor):
        # Split the input into three groups along the channel axis
        split_tensor = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutional layers with different kernel sizes
        conv1x1 = layers.SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(split_tensor[0])
        conv3x3 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(split_tensor[1])
        conv5x5 = layers.SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same')(split_tensor[2])
        
        # Concatenate the outputs from the three groups
        output_tensor = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
        
        return output_tensor
    
    split_conv_output = split_convolution(input_layer)
    batch_norm = layers.BatchNormalization()(split_conv_output)
    
    # Second block
    def block2(input_tensor):
        # Branch 1: 3x3 convolution
        conv3x3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_tensor)
        
        # Branch 2: series of layers
        conv1x1 = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(input_tensor)
        conv3x3_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv1x1)
        conv3x3_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv3x3_1)
        
        # Branch 3: max pooling
        maxpool = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        
        # Concatenate the outputs from all branches
        output_tensor = layers.Concatenate()([conv3x3, conv3x3_2, maxpool])
        
        return output_tensor
    
    block2_output = block2(batch_norm)
    batch_norm2 = layers.BatchNormalization()(block2_output)
    global_avg_pool = layers.GlobalAveragePooling2D()(batch_norm2)
    
    # Output layer
    output_layer = layers.Dense(units=10, activation='softmax')(global_avg_pool)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Usage
model = dl_model()
model.summary()