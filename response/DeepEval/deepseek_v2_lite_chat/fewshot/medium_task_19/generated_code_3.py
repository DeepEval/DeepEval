import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch: 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    # Second branch: 1x1 -> 3x3 convolution
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv2)
    # Third branch: 1x1 -> 5x5 convolution
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(conv3)
    # Fourth branch: 3x3 max pooling -> 1x1 convolution
    maxpool = MaxPooling2D(pool_size=(3, 3))(input_layer)
    maxpool = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(maxpool)
    
    # Concatenate the outputs from the four branches
    concat = Concatenate(axis=-1)([conv1, conv2, conv3, maxpool])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Build the model
model = dl_model()
model.summary()