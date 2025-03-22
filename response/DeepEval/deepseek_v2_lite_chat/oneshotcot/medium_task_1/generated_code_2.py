import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # MaxPooling layer
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Define the block
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block_output = block(maxpool)
    batch_norm = BatchNormalization()(block_output)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the constructed model
model = dl_model()
model.summary()