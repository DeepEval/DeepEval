import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Add, Activation
from keras.models import Model

def dl_model():
    # Initial convolution layer
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_shape[1:])
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # First parallel block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second parallel block
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Third parallel block
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Add outputs of parallel blocks
    x = Add()([x, initial_conv_output])  # assuming 'initial_conv_output' is the output of the initial convolution
    
    # Flatten the output for the fully connected layers
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output layer with Softmax activation
    output = Dense(num_classes, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=input_image, outputs=output)
    
    return model

# Assuming input_shape is (244, 244, 3) for CIFAR-10 and num_classes is 10
input_shape = (32, 32, 3)  # Adjust to fit the CIFAR-10 data
num_classes = 10

# Call the function to get the model
model = dl_model()