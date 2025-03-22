import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid')(path2)
    
    # Path 3: 1x1 convolution with two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid')(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid')(path3)
    
    # Path 4: Average pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(path4)
    
    # Concatenate the outputs of the paths
    concat_layer = Concatenate()([path1, path2, path3, path4])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concat_layer)
    
    # Flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()