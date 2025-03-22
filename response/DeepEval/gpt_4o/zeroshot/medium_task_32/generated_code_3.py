import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape for CIFAR-10 images (32x32x3)
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the last dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Depthwise Separable Convolution on each split with different kernel sizes
    conv1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
    conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
    conv5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
    
    # Concatenate the outputs
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification (CIFAR-10 has 10 classes)
    outputs = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example: Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()