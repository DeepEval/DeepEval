import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolutional layer with dropout
    conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(rate=0.3)(conv1x1_1)
    
    # Second 1x1 convolutional layer with dropout
    conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(dropout1)
    dropout2 = Dropout(rate=0.3)(conv1x1_2)
    
    # 3x1 convolutional layer with dropout
    conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(dropout2)
    dropout3 = Dropout(rate=0.3)(conv3x1)
    
    # 1x3 convolutional layer with dropout
    conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(dropout3)
    dropout4 = Dropout(rate=0.3)(conv1x3)
    
    # Restoring the number of channels to match input using 1x1 convolutional layer
    conv_restore_channels = Conv2D(filters=1, kernel_size=(1, 1), activation='relu', padding='same')(dropout4)
    
    # Adding the processed features with the original input
    added = Add()([input_layer, conv_restore_channels])
    
    # Flattening the output
    flatten_layer = Flatten()(added)
    
    # Fully connected layer to produce final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model