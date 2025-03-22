import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Initial 1x1 convolutional layer
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        # Branches for feature extraction, downsampling, and upsampling
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_tensor)
        branch2 = MaxPooling2D(pool_size=(2, 2))(branch1)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch2)
        branch4 = MaxPooling2D(pool_size=(2, 2))(branch3)
        branch5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch4)
        branch6 = UpSampling2D(size=(2, 2))(branch5)
        branch7 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch6)
        branch8 = UpSampling2D(size=(2, 2))(branch7)
        
        # Concatenate the outputs from all branches
        concat = Concatenate()([branch1, branch3, branch5, branch7, branch2, branch4, branch6, branch8])
        
        # Final 1x1 convolutional layer for the main path
        conv9 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
        
        # Fully connected layer for the main path
        flat = Flatten()(conv9)
        fc1 = Dense(units=512, activation='relu')(flat)
        fc2 = Dense(units=10, activation='softmax')(fc1)
        
        # Model for the main path
        main_model = keras.Model(inputs=input_layer, outputs=fc2)
        main_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return main_model

    # Branch path
    def branch_path(input_tensor):
        # Initial 1x1 convolutional layer
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        # Add padding to match the output shape of main path
        conv1 = ZeroPadding2D(padding=(0, 1))(conv1)
        # Additional convolution layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
        # UpSampling2D to match the input shape of the main path
        up_conv1 = UpSampling2D(size=(2, 2))(conv2)
        # Convolution layer to upscale the feature maps
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(up_conv1)
        # Add padding to match the output shape of main path
        conv3 = ZeroPadding2D(padding=(0, 1))(conv3)
        # Additional convolution layer
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv3)
        
        # Fully connected layers for the branch path
        branch_flat = Flatten()(conv4)
        fc1 = Dense(units=512, activation='relu')(branch_flat)
        fc2 = Dense(units=10, activation='softmax')(fc1)
        
        # Model for the branch path
        branch_model = keras.Model(inputs=input_tensor, outputs=fc2)
        branch_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return branch_model

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=[main_path(input_layer), branch_path(input_layer)])

    return model

# Create the model
model = dl_model()
model.summary()