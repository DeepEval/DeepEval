import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Define the first feature extraction block
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block1)
    
    # Define the second feature extraction block
    block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(block2)
    
    # Define the third feature extraction block
    block3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block3)
    
    # Concatenate the output of the feature extraction blocks
    concatenate_layer = Concatenate()([block1, block2, block3])
    
    # Define the flatten layer
    flatten_layer = Flatten()(concatenate_layer)
    
    # Define the first fully connected layer with dropout
    dense1 = Dense(units=1024, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    
    # Define the second fully connected layer with dropout
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    
    # Define the output layer with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(dropout2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model