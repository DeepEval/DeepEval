# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Activation, Concatenate
from tensorflow.keras.applications import CIFAR10DataGenerator
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
import tensorflow as tf

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape, name='input_layer')
    
    # Convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    
    # First parallel path: global average pooling followed by two fully connected layers
    avg_path = GlobalAveragePooling2D()(x)
    avg_path = Dense(64, activation='relu')(avg_path)
    avg_path = Dense(10, activation='softmax')(avg_path)
    
    # Second parallel path: global max pooling followed by two fully connected layers
    max_path = GlobalMaxPooling2D()(x)
    max_path = Dense(64, activation='relu')(max_path)
    max_path = Dense(10, activation='softmax')(max_path)
    
    # Add the outputs from the two paths
    add_path = Add()([avg_path, max_path])
    
    # Apply activation function to generate channel attention weights
    channel_attention = Activation('sigmoid')(add_path)
    
    # Apply channel attention weights to the original features
    x = Multiply()([x, channel_attention])
    
    # Separate average and max pooling operations to extract spatial features
    avg_pool = tf.reduce_mean(x, axis=[1, 2], name='avg_pool')
    max_pool = tf.reduce_max(x, axis=[1, 2], name='max_pool')
    
    # Concatenate the spatial features along the channel dimension
    x = Concatenate()([avg_pool, max_pool])
    
    # Combine the spatial and channel features through element-wise multiplication
    x = Multiply()([x, channel_attention])
    
    # Flatten the concatenated features
    x = tf.keras.layers.Flatten()(x)
    
    # Final fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])