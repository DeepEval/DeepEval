import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split1, split2, split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # First group remains unchanged
    unchanged = split1
    
    # Second group: 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split2)
    
    # Third group: additional 3x3 convolution
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split3)
    
    # Concatenate the outputs from the three groups
    concat = Concatenate(axis=-1)([unchanged, conv2, conv3])
    
    # Branch path: 1x1 convolutional layer
    branch = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    
    # Fuse the outputs from main and branch paths
    fused = keras.layers.Add()([concat, branch])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(fused)
    flat = Flatten()(bn)
    
    # Dense layers for final classification
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model