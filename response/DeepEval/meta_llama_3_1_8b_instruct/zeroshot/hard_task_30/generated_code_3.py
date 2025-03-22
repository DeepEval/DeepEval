import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the main path of the first block
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layers.Input(shape=input_shape))
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    
    # Define the branch path of the first block
    branch_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layers.Input(shape=input_shape))
    
    # Combine the main and branch paths through addition
    combined_path = layers.Add()([main_path, branch_path])
    
    # Define the second block with depthwise separable convolutional layers
    def separable_conv_block(x, filters, kernel_size):
        x = layers.SeparableConv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(combined_path)
    
    x1 = separable_conv_block(x[0], 32, (1, 1))
    x2 = separable_conv_block(x[1], 32, (3, 3))
    x3 = separable_conv_block(x[2], 32, (5, 5))
    
    x = layers.Concatenate(axis=-1)([x1, x2, x3])
    
    # Establish the model architecture
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=combined_path, outputs=x)
    
    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])