import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)
    
    # Split the input along the channel dimension into three groups
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply 1x1 convolutions to each group
    convs = [layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split) for split in splits]
    
    # Downsample each group with average pooling
    pooled = [layers.AveragePooling2D(pool_size=(2, 2))(conv) for conv in convs]
    
    # Concatenate the pooled feature maps along the channel dimension
    concatenated = layers.Concatenate(axis=-1)(pooled)
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = layers.Flatten()(concatenated)
    
    # Fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=dense2)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()