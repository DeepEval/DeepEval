import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv2)
    
    # Concatenate the outputs of convolutional layers
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()