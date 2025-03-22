from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Layer

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 64))
    
    # Main path
    x = inputs
    x = Conv2D(64, (1, 1), activation='relu')(x)  # 1x1 conv for dimensionality reduction
    x1 = Conv2D(128, (1, 1), activation='relu')(x)
    x2 = Conv2D(128, (3, 3), activation='relu')(x)
    x = Concatenate()([x1, x2])  # concatenate the outputs
    
    # Branch path
    x = Conv2D(128, (3, 3), activation='relu')(x)  # 3x3 conv to match channel count
    
    # Combine outputs from main and branch paths
    x = Concatenate()([x, x])
    
    # Flattening layer
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)  # Assuming 2 classes for classification
    
    # Model
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Build the model
model = dl_model()
model.summary()