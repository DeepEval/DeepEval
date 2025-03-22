from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    # Dimensionality reduction with 1x1 convolution
    main_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Parallel convolutional layers
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(main_path)
    conv3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)
    
    # Concatenate the outputs of the parallel layers
    main_path = Concatenate()([conv1x1, conv3x3])
    
    # Branch path
    branch_path = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    
    # Combine main path and branch path using addition
    combined = Add()([main_path, branch_path])
    
    # Flatten the output
    flat = Flatten()(combined)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for classification
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()