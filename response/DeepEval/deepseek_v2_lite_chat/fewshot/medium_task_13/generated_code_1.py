import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Define a function to apply the convolutional layers and max-pooling
    def add_conv_layers(x):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x
    
    # Apply the convolutional layers
    x = add_conv_layers(input_layer)
    
    # Repeat the convolutional layers with batch normalization
    x = add_conv_layers(x)
    
    # Flatten the output for the fully connected layers
    x = Flatten()(x)
    
    # Fully connected layers for classification
    output_layer = Dense(units=100, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(output_layer)
    
    # Return the constructed model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
return model