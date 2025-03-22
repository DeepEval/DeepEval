import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input shape
    num_classes = 10  # Number of classes in CIFAR-10

    model = tf.keras.models.Sequential([
        layers.Input(shape=input_shape),
        
        # Split input channels into three groups
        layers.Lambda(lambda x: tf.split(x, 3, axis=2)), 
        
        # Apply 1x1 convolutions to each group
        layers.Conv2D(input_shape[2] // 3, (1, 1), activation='relu'),  
        layers.Conv2D(input_shape[2] // 3, (1, 1), activation='relu'),  
        layers.Conv2D(input_shape[2] // 3, (1, 1), activation='relu'), 

        # Average pooling for downsampling
        layers.AveragePooling2D((2, 2)),
        
        # Concatenate feature maps from each group
        layers.Concatenate(axis=2),
        
        # Flatten and classify
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model