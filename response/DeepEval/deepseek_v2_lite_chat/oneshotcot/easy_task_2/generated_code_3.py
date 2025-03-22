import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     
    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # First feature extraction layer
    conv1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv1)
    
    # Second feature extraction layer
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv2)
    
    # Add three convolutional layers
    def block(input_tensor):
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv3)
        return avg_pool3
    
    conv3_output = block(avg_pool2)
    avg_pool4 = block(conv3_output)
    conv4_output = block(avg_pool4)
    
    # Concatenate the outputs of the three convolutional paths
    concat_layer = Concatenate()( [conv3_output, conv4_output, conv4_output, conv4_output] )
    
    # Batch normalization
    batch_norm = BatchNormalization()(concat_layer)
    
    # Flatten
    flatten = Flatten()(batch_norm)
    
    # Two fully connected layers with dropout
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    
    # Output layer
    output_layer = Dense(units=1000, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model