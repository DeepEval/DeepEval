import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 color channel
    inputs = layers.Input(shape=input_shape)
    
    # First block: average pooling layers
    pool1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    pool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    
    # Flatten the pooled outputs
    flat1 = layers.Flatten()(pool1)
    flat2 = layers.Flatten()(pool2)
    flat3 = layers.Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concatenated = layers.Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer
    fc1 = layers.Dense(128, activation='relu')(concatenated)
    
    # Reshape output into a 4D tensor
    reshaped = layers.Reshape((1, 1, 128))(fc1)
    
    # Second block: depthwise separable convolutions
    split_tensors = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)
    
    # Depthwise separable convolutions with different kernel sizes
    conv1 = layers.DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    conv2 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    conv3 = layers.DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
    conv4 = layers.DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split_tensors[3])
    
    # Concatenate the convolution outputs
    conv_concatenated = layers.Concatenate()([conv1, conv2, conv3, conv4])
    
    # Flatten the concatenated output
    flattened_output = layers.Flatten()(conv_concatenated)
    
    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened_output)  # 10 classes for MNIST
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()  # To display the model's architecture