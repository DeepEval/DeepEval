import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    def block1(input_tensor):
        maxpool1 = layers.MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        maxpool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        maxpool3 = layers.MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Flatten the pooling outputs
        flat1 = layers.Flatten()(maxpool1)
        flat2 = layers.Flatten()(maxpool2)
        flat3 = layers.Flatten()(maxpool3)

        # Concatenate the flattened outputs and apply dropout
        concatenated = layers.Concatenate()([flat1, flat2, flat3])
        dropout = layers.Dropout(0.2)(concatenated)

        # Reshape the output to match the input of block2
        reshaped = layers.Reshape((1024,))(dropout)

        return reshaped

    # Define the second block
    def block2(input_tensor):
        # Split the input into four groups
        split = tf.split(input_tensor, num_or_size_splits=4, axis=-1)

        # Process each group using a separable convolution
        conv1 = layers.SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(split[0])
        conv2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(split[1])
        conv3 = layers.SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(split[2])
        conv4 = layers.SeparableConv2D(filters=64, kernel_size=(7, 7), activation='relu')(split[3])

        # Concatenate the outputs from the four groups
        concatenated = layers.Concatenate()([conv1, conv2, conv3, conv4])

        # Flatten the output
        flat = layers.Flatten()(concatenated)

        return flat

    # Create the model
    input_layer = layers.Input(shape=input_shape)
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    dense = layers.Dense(units=10, activation='softmax')(block2_output)
    
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model