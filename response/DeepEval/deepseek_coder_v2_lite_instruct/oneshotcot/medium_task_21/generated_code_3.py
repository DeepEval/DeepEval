import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions (equivalent to one 5x5 convolution)
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    
    # Concatenate the outputs of the four branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Batch Normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Dropout to mitigate overfitting
    dropout1 = Dropout(0.5)(flatten_layer)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout2)
    dropout3 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout3)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model