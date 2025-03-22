import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolution to reduce dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    
    # 1x1 convolution to restore dimensionality
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)
    
    # Flatten the output
    flatten = Flatten()(conv3)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()