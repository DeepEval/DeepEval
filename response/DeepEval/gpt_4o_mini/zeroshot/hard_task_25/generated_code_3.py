import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Main path
    x = layers.Conv2D(64, (1, 1), activation='relu')(input_layer)
    
    # Splitting into three branches
    # Branch 1
    branch1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    # Branch 2
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3
    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(branch3)

    # Concatenate outputs of all branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])
    
    # 1x1 Convolution to form the main path output
    main_path_output = layers.Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Branch path
    branch_path_output = layers.Conv2D(64, (1, 1), activation='relu')(input_layer)
    
    # Fuse main path and branch path outputs through addition
    fused_output = layers.Add()([main_path_output, branch_path_output])
    
    # Global average pooling before the fully connected layer
    x = layers.GlobalAveragePooling2D()(fused_output)
    
    # Fully connected layer for 10-class classification
    output_layer = layers.Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()  # To visualize the model architecture