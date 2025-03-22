import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: 1x1 convolution followed by average pooling
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1 = AveragePooling2D(pool_size=(2, 2))(path1)
    
    # Path 2: 1x1 convolution followed by a 1x1 convolution
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(path2)
    
    # Path 3: 1x1 convolution followed by two parallel convolutions
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='valid')(path3)
    path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid')(path3)
    path3 = Concatenate(axis=3)([path3, path3_1x3, path3_3x1])
    
    # Path 4: 1x1 convolution followed by a 3x3 convolution
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(path4)
    
    # Concatenate all paths
    concat_layer = Concatenate(axis=3)([path1, path2, path3, path4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()