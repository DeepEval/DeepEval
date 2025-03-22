from keras.datasets import cifar10
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels

    # Input layers
    input_layer = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 3
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Concatenate the outputs of each block
    concat = Concatenate(axis=-1)([x, x])
    
    # Flatten the concatenated output
    flat = Flatten()(concat)
    
    # Fully connected layers
    output_layer = Dense(10, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()