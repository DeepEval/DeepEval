from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # 1x1 Convolution on input
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Branch 2: Average Pooling, 3x3 Convolution, Transposed Convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(branch2)
    
    # Branch 3: Average Pooling, 3x3 Convolution, Transposed Convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    branch3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(branch3)
    
    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 Convolution after concatenation
    main_output = Conv2D(64, (1, 1), activation='relu', padding='same')(concatenated)
    
    # Branch path
    branch_output = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    
    # Fuse main path and branch path outputs
    fused_output = Add()([main_output, branch_output])
    
    # Fully connected layer for classification
    x = Flatten()(fused_output)
    final_output = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=final_output)
    
    return model

# Use the function to create the model
model = dl_model()
model.summary()