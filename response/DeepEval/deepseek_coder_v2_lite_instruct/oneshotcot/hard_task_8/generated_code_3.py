import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, DepthwiseConv2D, Permute, Reshape
from keras.models import Model

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    primary_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', depthwise_mode=True)(primary_path)
    primary_path = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(primary_path)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', depthwise_mode=True)(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch_path)
    
    # Concatenate features from both paths
    concatenated_features = Concatenate(axis=-1)([primary_path, branch_path])
    
    # Block 2
    # Reshape features from Block 1 to target shape
    reshaped_features = Reshape((-1, 28, 28, 1))(concatenated_features)
    
    # Permute dimensions to shuffle channels
    permuted_features = Permute((1, 2, 4, 3))(reshaped_features)
    
    # Flatten the features
    flattened_features = Flatten()(permuted_features)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and return the model
model = dl_model()
model.summary()