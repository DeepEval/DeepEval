from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Concatenate the input with the output of the first convolutional layer
    concat1 = Concatenate()([inputs, conv1])
    
    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(concat1)
    
    # Concatenate the result with the output of the previous concat
    concat2 = Concatenate()([concat1, conv2])
    
    # Third convolutional layer
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(concat2)
    
    # Concatenate the result with the output of the previous concat
    concat3 = Concatenate()([concat2, conv3])
    
    # Flatten the output
    flat = Flatten()(concat3)
    
    # First fully connected layer
    dense1 = Dense(512, activation='relu')(flat)
    
    # Output layer for 10 classes
    outputs = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()