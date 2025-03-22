import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction using fully connected layers
    # Global average pooling to generate weights
    avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers
    dense1 = Dense(units=1024, activation='relu')(avg_pool)
    dense2 = Dense(units=512, activation='relu')(dense1)
    
    # Reshape the dense layer output to match the input shape
    reshaped_dense = keras.layers.Reshape((-1, 512))(dense2)
    
    # Multiply with the original input to get the weighted features
    weighted_features = keras.layers.Multiply()([input_layer, reshaped_dense])
    
    # Block 2: Feature extraction using convolutional layers and max pooling
    # First convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    # Max pooling layer
    maxpool = MaxPooling2D(pool_size=(3, 3))(conv2)
    
    # Connect Block 1 and Block 2
    branch = weighted_features  # Directly use the weighted features as the branch
    main_output = maxpool  # Output from the main path
    
    # Addition to fuse the branch and the main output
    fused_output = Add()([branch, main_output])
    
    # Classification using fully connected layers
    dense3 = Dense(units=256, activation='relu')(fused_output)
    dense4 = Dense(units=128, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()