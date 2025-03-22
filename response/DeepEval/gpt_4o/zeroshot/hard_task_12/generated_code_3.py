from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    # 1x1 convolution for dimensionality reduction
    main_path = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    
    # Parallel convolutions
    conv1x1 = Conv2D(64, (1, 1), activation='relu', padding='same')(main_path)
    conv3x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(main_path)
    
    # Concatenate the outputs of the parallel convolutions
    main_path = Concatenate()([conv1x1, conv3x3])
    
    # Branch path
    branch_path = Conv2D(128, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Combine main and branch paths
    combined = Add()([main_path, branch_path])
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(combined)
    dense1 = Dense(256, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)  # Assuming 10 classes for classification
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()