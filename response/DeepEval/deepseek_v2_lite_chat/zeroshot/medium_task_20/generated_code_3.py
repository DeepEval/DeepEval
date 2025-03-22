import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import Lambda
from keras.optimizers import Adam

# Number of classes
num_classes = 10

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First convolutional path
    conv1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Second convolutional path
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    
    # Third convolutional path
    conv3 = Conv2D(64, (3, 3), activation='relu')(conv1)
    
    # Fourth convolutional path
    conv4 = MaxPooling2D(pool_size=(3, 3))(inputs)
    conv4 = Conv2D(64, (1, 1), activation='relu')(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu')(conv4)
    
    # Concatenate the outputs from all paths
    concat = concatenate([conv2, conv3, conv4])
    
    # Flatten and pass through a dense layer
    dense1 = Flatten()(concat)
    dense1 = Dense(128, activation='relu')(dense1)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Optional: Create an instance of the model
model = dl_model()
model.summary()