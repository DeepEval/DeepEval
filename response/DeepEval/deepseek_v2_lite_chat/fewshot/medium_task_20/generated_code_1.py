import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, AveragePooling2D, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional path
    conv_path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv_path1 = MaxPooling2D(pool_size=(2, 2))(conv_path1)
    
    # Second convolutional path
    conv_path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_path2 = MaxPooling2D(pool_size=(2, 2))(conv_path2)
    
    # Third convolutional path
    conv_path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_path3 = MaxPooling2D(pool_size=(2, 2))(conv_path3)
    
    # Fourth convolutional path
    conv_path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    conv_path4 = AveragePooling2D(pool_size=(2, 2))(conv_path4)
    conv_path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv_path4)
    
    # Concatenate and flatten outputs
    concatenated = Concatenate()([conv_path1, conv_path2, conv_path3, conv_path4])
    flattened = Flatten()(concatenated)
    
    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flattened)
    
    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()