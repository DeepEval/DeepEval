import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    def block(input_tensor):
        # First path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        norm1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)
        
        # Second path
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        norm2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(norm2)
        
        # Third path
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
        norm3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(norm3)
        
        # Concatenate all paths
        concatenated = Concatenate(axis=-1)([norm1, norm2, norm3, pool3])
        
        return concatenated
    
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Construct the first block
    first_block = block(inputs)
    
    # Additional blocks (you can add as many as needed)
    second_block = block(first_block)
    third_block = block(second_block)
    
    # Flatten and dense layers
    flat_layer = Flatten()(third_block)
    dense1 = Dense(units=256, activation='relu')(flat_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=inputs, outputs=output_layer)
    
    return model