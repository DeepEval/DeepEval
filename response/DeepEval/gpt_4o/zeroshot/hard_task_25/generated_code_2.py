from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    
    # Branch 2: Average pooling -> 3x3 Convolution -> Transpose Convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), padding='same')(main_path)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same')(branch2)
    
    # Branch 3: Average pooling -> 3x3 Convolution -> Transpose Convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), padding='same')(main_path)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same')(branch3)
    
    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 convolution on concatenated features
    main_path_output = Conv2D(32, (1, 1), activation='relu', padding='same')(concatenated)
    
    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    
    # Fuse main path output and branch path output
    fused_output = Add()([main_path_output, branch_path])
    
    # Fully connected layer for classification
    flattened = Flatten()(fused_output)
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of creating and summarizing the model
model = dl_model()
model.summary()