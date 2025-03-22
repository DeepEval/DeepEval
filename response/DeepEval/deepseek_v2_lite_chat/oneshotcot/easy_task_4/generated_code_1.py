import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block of convolutional layers
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block1)
    conv_block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_block1)
    
    # Second block of convolutional layers
    conv_block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block1)
    conv_block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block2)
    conv_block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_block2)
    
    # Concatenate the feature maps from both blocks
    concat = Concatenate()(list(conv_block2.output))
    
    # Flatten the concatenated feature maps
    flat = Flatten()(concat)
    
    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()