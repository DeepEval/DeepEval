from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

def custom_block(input_tensor, channels):
    # Global Average Pooling
    x = GlobalAveragePooling2D()(input_tensor)
    
    # Fully connected layers to produce weights
    x = Dense(channels // 2, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    
    # Reshape the weights to match input shape
    x = Reshape((1, 1, channels))(x)
    
    # Element-wise multiply with the input tensor
    x = Multiply()([input_tensor, x])
    
    return x

def dl_model():
    # Define input shape for CIFAR-10
    input_shape = (32, 32, 3)
    num_classes = 10
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First branch
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = custom_block(x1, 32)
    
    # Second branch
    x2 = Conv2D(32, (3, 3), padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = custom_block(x2, 32)
    
    # Concatenate the outputs from both branches
    concatenated = Concatenate()([x1, x2])
    
    # Flatten and final fully connected layer for classification
    x = Flatten()(concatenated)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model