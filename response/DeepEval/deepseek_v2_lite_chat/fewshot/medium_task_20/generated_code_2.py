import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    
    # Second convolutional path
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_1)
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    
    # Third convolutional path
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
    
    # Fourth convolutional path
    conv4_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(input_layer)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    
    # Concatenate the outputs from each path
    concat = Concatenate()([pool1, pool2, pool3, pool4])
    
    # Flatten the concatenated output
    flatten = Flatten()(concat)
    
    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()