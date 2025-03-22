import keras
from keras.layers import Input, Conv2D, AveragePooling2D, ZeroPadding2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Conv2DTranspose, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # 1x1 convolutional layer
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Split into three branches
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        branch2 = AveragePooling2D(pool_size=(1, 1), padding='valid')(conv1)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        transpose2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)
        transpose2 = BatchNormalization()(transpose2)
        transpose2 = Activation('relu')(transpose2)
        transpose2 = UpSampling2D(size=2)(transpose2)
        
        # Concatenate all branches
        concat = Concatenate()([branch1, branch2, transpose2])
        # Additional 1x1 convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
        
        # Branch path
        branch_input = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 convolutional layer to match channels
        branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch_input)
        # Add to main path output
        fused = keras.layers.add([conv2, branch_conv1])
        
        # Flatten and dense layers
        flat = Flatten()(fused)
        dense1 = Dense(units=128, activation='relu')(flat)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    
    # Construct and return the model
    return main_path(input_layer)

# Build the model
model = dl_model()
model.summary()