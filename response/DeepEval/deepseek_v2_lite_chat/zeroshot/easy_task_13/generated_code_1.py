import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.layers import Dropout, LeakyReLU

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)
    
    # First 1x1 convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    conv1 = Dropout(0.2)(conv1)
    
    # Second 1x1 convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)
    
    # 3x1 convolutional layer
    conv3 = Conv2D(64, (3, 1), padding='valid', strides=(1, 3))(conv2)
    conv3 = Dropout(0.2)(conv3)
    
    # 1x3 convolutional layer
    conv4 = Conv2D(64, (1, 3), padding='valid', strides=(1, 1))(conv3)
    conv4 = Dropout(0.2)(conv4)
    
    # Restore channels to match input
    conv5 = Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(conv4)
    
    # Flatten layer
    flat = Flatten()(conv5)
    
    # Fully connected layer
    dense = Dense(128, activation='relu')(flat)
    
    # Output layer
    output = Dense(10, activation='softmax')(dense)
    
    # Model construction
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Get the model
model = dl_model()
model.summary()