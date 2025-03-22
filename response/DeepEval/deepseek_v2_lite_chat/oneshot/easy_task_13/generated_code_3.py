import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Second 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv1)
    
    # 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), padding='valid')(input_layer)
    
    # 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), padding='valid')(input_layer)
    
    # 1x1 convolutional layer to adjust channels (assuming input has a channel dimension)
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Flatten the input for concatenation
    flatten = Flatten()(conv5)
    
    # Concatenate all convolutional layers
    concatenated = Concatenate()([conv2, conv3, conv4, flatten])
    
    # Add the concatenated features to the original input
    output = Add()([input_layer, concatenated])
    
    # Apply dropout for regularization
    dropout = keras.layers.Dropout(0.5)(output)
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(dropout)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and print the model
model = dl_model()
model.summary()