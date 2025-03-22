import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply different convolutional kernels to each group
    conv1x1 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    conv3x3 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    conv5x5 = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
    
    # Concatenate the outputs from the three groups
    concatenated = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()