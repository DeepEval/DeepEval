import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Parallel branches
    # 1x1 convolution branch
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # 3x3 convolution branch
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # 5x5 convolution branch
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_layer)
    
    # 3x3 max pooling branch
    pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=1)(input_layer)
    
    # Concatenate the outputs of the branches
    concatenated = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, pool3x3])
    
    # Second block: Global Average Pooling and Fully Connected Layers
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(concatenated)
    
    # Fully Connected Layers
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for CIFAR-10
    
    # Reshape weights to match the input's shape and multiply with the input feature map
    weights = Reshape((1, 1, 32))(fc2)  # Assuming 32 channels as per the example
    weighted_input = tf.multiply(input_layer, weights)
    
    # Final fully connected layer
    final_fc = Dense(10, activation='softmax')(weighted_input)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=final_fc)
    
    return model

# Example usage
model = dl_model()
model.summary()