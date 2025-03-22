import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Dual-path structure
    def block1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        
        # Branch path
        branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        
        # Combine main and branch paths
        combined = Add()([conv2, branch])
        
        return combined
    
    # Block 2: Three groups of depthwise separable convolutional layers
    def block2(input_tensor):
        # Split input into three groups
        split1 = Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)
        
        # Depthwise separable convolutional layers for each group
        conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[0])
        conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(split1[1])
        conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(split1[2])
        
        # Concatenate outputs
        concatenated = Concatenate(axis=1)([conv1x1, conv3x3, conv5x5])
        
        # Flatten and fully connected layers
        flatten = Flatten()(concatenated)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=block2(block1(input_layer)))
    
    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()