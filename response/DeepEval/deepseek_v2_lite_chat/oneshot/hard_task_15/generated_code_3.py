import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    x = GlobalAveragePooling2D()(inputs)  # Global average pooling
    x = Dense(units=128, activation='relu')(x)  # Fully connected layer
    x = Dense(units=64, activation='relu')(x)   # Fully connected layer
    
    # Branch path
    branch_output = inputs  # No modifications in the branch path

    # Concatenate outputs from both paths
    combined_output = Concatenate()([x, branch_output])
    
    # Additional fully connected layers
    x = Dense(units=128, activation='relu')(combined_output)
    x = Dense(units=64, activation='relu')(x)
    outputs = Dense(units=10, activation='softmax')(x)  # Assuming 10 classes for CIFAR-10
    
    # Model construction
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build and print the model
model = dl_model()
model.summary()