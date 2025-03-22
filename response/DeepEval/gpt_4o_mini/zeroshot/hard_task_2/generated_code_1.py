import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32, 3 channel)
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_channels = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the function for the convolutional sequence
    def conv_sequence(x):
        x = layers.Conv2D(16, (1, 1), activation='relu')(x)  # 1x1 Conv
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)  # 3x3 Conv
        x = layers.Conv2D(16, (1, 1), activation='relu')(x)  # 1x1 Conv
        return x
    
    # Apply the convolutional sequence to each split channel
    group_outputs = [conv_sequence(channel) for channel in split_channels]
    
    # Combine the outputs of the three groups using addition
    main_path = layers.Add()(group_outputs)
    
    # Fuse the main path with the original input layer
    fused_output = layers.Add()([main_path, input_layer])
    
    # Flatten the features
    flattened_output = layers.Flatten()(fused_output)
    
    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flattened_output)  # 10 classes for MNIST
    
    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()