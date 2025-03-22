import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Convolutional layer followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Additional convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool1)
    
    # Flatten the feature maps
    flat_layer = Flatten()(conv2)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flat_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])