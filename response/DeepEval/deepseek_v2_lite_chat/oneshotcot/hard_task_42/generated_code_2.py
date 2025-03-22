import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Three parallel paths with max pooling of different scales
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        # Flatten and apply dropout
        flat_tensor1 = Flatten()(path1)
        flat_tensor2 = Flatten()(path2)
        flat_tensor3 = Flatten()(path3)
        dropout1 = Dropout(0.5)(flat_tensor1)
        dropout2 = Dropout(0.5)(flat_tensor2)
        dropout3 = Dropout(0.5)(flat_tensor3)
        # Concatenate all paths
        concat_tensor = Concatenate()(outputs=[dropout1, dropout2, dropout3])
        
        return concat_tensor
    
    # Block 2
    def block2(input_tensor):
        # Four parallel paths with diverse convolution and pooling
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path1)
        path2 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path2)
        path3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path3)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
        
        # Concatenate all paths along the channel dimension
        concat_tensor = Concatenate(axis=-1)(outputs=[path1, path2, path3, path4])
        
        # Fully connected layers for final classification
        dense1 = Dense(units=256, activation='relu')(concat_tensor)
        dense2 = Dense(units=128, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Model construction
    block1_output = block1(input_tensor=input_layer)
    output1 = block2(input_tensor=block1_output)
    
    model = Model(inputs=input_layer, outputs=output1)
    
    return model

# Construct the model
model = dl_model()

# Display the model summary
model.summary()