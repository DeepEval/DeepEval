import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Initial 1x1 convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 1: 3x3 convolutional layer for feature extraction
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        # Branch 2 and 3: Downsample and upsample feature maps using max pooling
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        upconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv3)
        merge1 = Concatenate()([upconv1, conv2, conv2, conv2])  # Upsample in each direction to match channel count
        
        # 1x1 convolutional layer to reduce dimensionality
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merge1)
        
        # Flatten and fully connected layers
        flatten = Flatten()(conv4)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        # Model
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model
    
    # Branch path
    branch_input = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    branch_model = main_path(branch_input)
    
    # Concatenate the outputs from the main and branch paths
    concat_layer = Concatenate()([model.output, branch_model.output])
    
    # Additional 1x1 convolutional layer to match the channel count of the main path
    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    
    # Final fully connected layers
    dense3 = Dense(units=128, activation='relu')(conv5)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Model
    model = keras.Model(inputs=[model.input, branch_model.input], outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])