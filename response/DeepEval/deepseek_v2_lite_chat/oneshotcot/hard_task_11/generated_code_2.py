import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    def main_path(input_tensor):
        # 1x1 convolution
        conv1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        # 1x1, 1x3, and 3x1 convolutions
        conv2 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        # Add the outputs of the paths
        concat = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
        # Another 1x1 convolution
        conv5 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(concat)
        # Output layer
        output = Conv2D(filters=10, kernel_size=1, strides=1, padding='same', activation='softmax')(conv5)
        
        return output
    
    # Direct connection pathway
    def direct_path(input_tensor):
        # 1x1 convolution
        conv1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        # Output layer
        output = Conv2D(filters=10, kernel_size=1, strides=1, padding='same', activation='softmax')(conv1)
        
        return output
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=main_path(direct_path(input_layer)))
    
    return model

# Build the model
model = dl_model()
model.summary()