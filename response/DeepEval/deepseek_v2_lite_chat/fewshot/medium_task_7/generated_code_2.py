import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    
    # Add the outputs of the first three convolutional layers
    add_layer = Add()([conv3, conv2, conv1])
    
    # Separate convolutional layer for direct input processing
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Flatten and pass through two fully connected layers for classification
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Building and compiling the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

return model