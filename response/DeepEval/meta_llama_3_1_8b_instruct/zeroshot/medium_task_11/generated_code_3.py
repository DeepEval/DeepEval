# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by processing the input through an initial convolutional layer. 
    It then extracts features along the channel dimension using two parallel paths: 
    one with global average pooling followed by two fully connected layers, and the other with global max pooling followed by two fully connected layers.
    
    The outputs from these two paths are added and passed through an activation function to generate channel attention weights, 
    which are then applied to the original features via element-wise multiplication.
    
    Next, the processed features undergo separate average and max pooling operations to extract spatial features, 
    which are concatenated along the channel dimension to form a fused feature map.
    
    Finally, the spatial features are combined with the channel features through element-wise multiplication, 
    flattened, and fed into a fully connected layer to produce the final output.
    
    Returns:
    model: A Keras model instance representing the deep learning model for image classification.
    """

    # Define the input shape of the model, which is the shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create a Keras model instance
    model = keras.Model()
    
    # Add an initial convolutional layer with 32 filters, kernel size 3, and ReLU activation
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(keras.Input(shape=input_shape))
    
    # Split the output into two parallel paths
    path1 = layers.GlobalAveragePooling2D()(x)
    path2 = layers.GlobalMaxPooling2D()(x)
    
    # Add two fully connected layers to each path with 128 units and ReLU activation
    path1 = layers.Dense(128, activation='relu')(path1)
    path2 = layers.Dense(128, activation='relu')(path2)
    
    # Concatenate the outputs of the two paths along the channel dimension
    path_concat = layers.Concatenate()([path1, path2])
    
    # Add a channel attention layer
    channel_attention = layers.Add()([path1, path2])
    channel_attention = layers.Activation('relu')(channel_attention)
    channel_attention = layers.Lambda(lambda x: tf.nn.sigmoid(x))(channel_attention)
    channel_attention = layers.Multiply()([x, channel_attention])
    
    # Merge the channel attention output with the original features
    merged = layers.Add()([channel_attention, x])
    
    # Add average and max pooling operations to extract spatial features
    avg_pool = layers.GlobalAveragePooling2D()(merged)
    max_pool = layers.GlobalMaxPooling2D()(merged)
    
    # Concatenate the spatial features along the channel dimension
    spatial_features = layers.Concatenate()([avg_pool, max_pool])
    
    # Merge the spatial features with the channel features through element-wise multiplication
    fused_features = layers.Multiply()([channel_attention, spatial_features])
    
    # Flatten the fused features and add a fully connected layer to produce the final output
    output = layers.Flatten()(fused_features)
    output = layers.Dense(10, activation='softmax')(output)
    
    # Add the output layer to the model
    model = keras.Model(inputs=keras.Input(shape=input_shape), outputs=output)
    
    return model

# Test the function
model = dl_model()
model.summary()