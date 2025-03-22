import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    # First block: Convolution + Dropout
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_path = Dropout(0.25)(main_path)
    
    # Second block: Convolution + Dropout
    main_path = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(main_path)
    main_path = Dropout(0.25)(main_path)
    
    # Additional convolution layer to restore the number of channels
    main_path = Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')(main_path)
    
    # Branch path
    branch_path = input_layer  # Direct connection to input
    
    # Combine both paths
    combined = Add()([main_path, branch_path])
    
    # Flatten and final dense layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model