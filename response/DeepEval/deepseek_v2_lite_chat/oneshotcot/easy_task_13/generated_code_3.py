import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    # Second 1x1 convolutional layer to match the number of input channels
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # 3x1 convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 1), padding='valid')(conv1)
    # 1x3 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), padding='valid')(conv2)
    
    # Dropout layer to mitigate overfitting
    dropout1 = Dropout(0.25)(conv3)
    dropout2 = Dropout(0.25)(conv4)
    
    # Concatenate the outputs of the first and second 1x1 convolutions
    concat = Concatenate(axis=-1)([conv1, dropout1, conv2, dropout2])
    
    # First batch normalization and flattening
    batch_norm1 = BatchNormalization()(concat)
    flatten_layer = Flatten()(batch_norm1)
    
    # Fully connected layer for the first time
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Third fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and print the model
model = dl_model()
model.summary()