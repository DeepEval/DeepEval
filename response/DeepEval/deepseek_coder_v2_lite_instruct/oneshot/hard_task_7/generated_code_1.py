import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Split the input into two groups
        split1, split2 = tf.split(input_tensor, num_or_size_splits=2, axis=-1)
        
        # First group operations
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(split1)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Second group remains unchanged
        
        # Merge outputs
        output_tensor = Concatenate()([path1, path3])
        
        return output_tensor
    
    block1_output = block1(conv1)
    batch_norm1 = BatchNormalization()(block1_output)
    
    # Block 2
    def block2(input_tensor):
        # Get the shape of the input
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        
        # Reshape the input into four groups
        reshaped = Reshape((height, width, 2, channels // 2))(input_tensor)
        
        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Flatten the input
        flattened = Flatten()(permuted)
        
        return flattened
    
    block2_output = block2(batch_norm1)
    dense_output = Dense(units=10, activation='softmax')(block2_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_output)
    
    return model

# Example usage
model = dl_model()
model.summary()