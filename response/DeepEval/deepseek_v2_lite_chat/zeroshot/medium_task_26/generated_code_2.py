from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 64))
    
    # Channel Compression with 1x1 Convolution
    x = Conv2D(16, (1, 1), activation='relu')(inputs)
    
    # Feature Expansion
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Concatenate the results
    x = Concatenate()([x, x])
    
    # Flatten the feature map
    x = Flatten()(x)
    
    # Fully Connected Layers
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # Assuming 10 classes
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()