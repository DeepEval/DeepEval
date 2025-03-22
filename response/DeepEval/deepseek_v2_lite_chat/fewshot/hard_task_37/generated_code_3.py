import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block for the first branch
    def block(input_tensor):
        # Convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        # Convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        # Convolutional layer
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        # Max pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        return pool1
    
    # Block for the second branch
    def block(input_tensor):
        # Convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
        # Max pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # Convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(pool1)
        # Convolutional layer
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(conv2)
        # Add main paths
        add1 = Add()([block(input_tensor), input_tensor])
        # Add branch path
        add2 = Add()([block(input_tensor), input_tensor])
        return add1, add2
    
    # Split the input into two paths
    branch1, branch2 = block(input_layer)
    
    # Combine the outputs from both branches using addition
    combined = Add()([branch1, branch2])
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()