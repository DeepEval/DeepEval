import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dropout, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    # Three parallel paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input_layer)
    # Max pooling layers with different pooling windows and strides
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path1)
    max_pool2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(path2)
    max_pool3 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(path3)
    # Flatten and dropout
    flatten1 = Flatten()(max_pool1)
    flatten2 = Flatten()(max_pool2)
    flatten3 = Flatten()(max_pool3)
    dropout1 = Dropout(0.2)(flatten1)
    dropout2 = Dropout(0.2)(flatten2)
    dropout3 = Dropout(0.2)(flatten3)
    # Concatenate the outputs
    concatenated = Concatenate()([dropout1, dropout2, dropout3])
    
    # Block 2
    # Four parallel paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(7, 7), strides=(7, 7), padding='same', activation='relu')(input_layer)
    # Concatenate the outputs along the channel dimension
    concatenated = Concatenate(axis=3)([path1, path2, path3, path4])
    # Flatten and dropout
    flattened = Flatten()(concatenated)
    dropout = Dropout(0.2)(flattened)
    
    # Final classification
    output = Dense(units=10, activation='softmax')(dropout)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model