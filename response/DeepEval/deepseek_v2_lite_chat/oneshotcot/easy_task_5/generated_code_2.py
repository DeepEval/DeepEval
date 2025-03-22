import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolution for dimensionality reduction
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    
    # 1x1 convolution to restore dimensionality
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv2)
    
    # Flatten the output
    flatten = Flatten()(conv3)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # Output layer with 10 neurons

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()