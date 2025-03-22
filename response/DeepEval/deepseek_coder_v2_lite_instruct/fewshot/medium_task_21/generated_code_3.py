import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_1)
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions (equivalent to one 5x5 convolution)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv3)
    
    # Branch 4: Average pooling followed by 1x1 convolution
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([conv1, conv2, conv3, conv4])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Apply dropout to mitigate overfitting
    dropout1 = Dropout(0.5)(flattened)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout2)
    dropout3 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout3)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model