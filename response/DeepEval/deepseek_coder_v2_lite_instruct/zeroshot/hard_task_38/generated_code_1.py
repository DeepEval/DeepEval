import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the first pathway
    def conv_block(x):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Define the second pathway
    def conv_block_2(x):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # Create the first pathway
    input_layer = Input(shape=input_shape)
    x1 = conv_block(input_layer)
    x1 = conv_block(x1)
    x1 = conv_block(x1)
    
    # Create the second pathway
    x2 = conv_block_2(input_layer)
    x2 = conv_block_2(x2)
    x2 = conv_block_2(x2)
    
    # Merge the outputs of both pathways
    merged = Concatenate(axis=-1)([x1, x2])
    
    # Flatten the merged output
    flatten = tf.keras.layers.Flatten()(merged)
    
    # Add fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()