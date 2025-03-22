from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input with shape (32, 32, 64)
    input_layer = Input(shape=(32, 32, 64))
    
    # Compress the input channels using a 1x1 convolution
    compressed = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # First parallel branch with 1x1 convolution
    conv1x1 = Conv2D(64, (1, 1), activation='relu')(compressed)
    
    # Second parallel branch with 3x3 convolution
    conv3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(compressed)
    
    # Concatenate the results of both branches
    concatenated = Concatenate()([conv1x1, conv3x3])
    
    # Flatten the concatenated feature map
    flat = Flatten()(concatenated)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flat)
    fc2 = Dense(10, activation='softmax')(fc1)  # Adjust the number of units according to the number of classes
    
    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)
    
    return model

# Example of creating the model
# model = dl_model()
# model.summary()