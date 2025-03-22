import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input image into three groups along the channel dimension
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply different convolutional kernels to each group
    conv1x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_layers[0])
    conv3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_layers[1])
    conv5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_layers[2])
    
    # Concatenate the outputs from these groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the fused features and pass them through two fully connected layers for classification
    flattened = Flatten()(concatenated)
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # 10 classes for CIFAR-10
    
    # Construct the model
    model = Model(inputs=inputs, outputs=fc2)
    
    return model

# Example usage
model = dl_model()
model.summary()