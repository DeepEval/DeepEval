import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolution to reduce dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # Restore dimensionality with another 1x1 convolution
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the output
    flatten = Flatten()(conv3)
    
    # Fully connected layer
    dense = Dense(units=10, activation='softmax')(flatten)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=dense)
    
    return model

# Create the model
model = dl_model()
model.summary()