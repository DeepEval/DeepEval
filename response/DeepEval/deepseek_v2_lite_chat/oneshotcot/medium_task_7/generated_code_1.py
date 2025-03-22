import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Convolutional layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Convolutional layer 3
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    
    # Process input directly
    direct_conv = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Add the outputs from all convolutional layers
    added_outputs = Add()([conv3, direct_conv])
    
    # Batch normalization
    batch_norm = BatchNormalization()(added_outputs)
    
    # Flatten layer
    flatten = Flatten()(batch_norm)
    
    # Fully connected layer 1
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Fully connected layer 2
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()