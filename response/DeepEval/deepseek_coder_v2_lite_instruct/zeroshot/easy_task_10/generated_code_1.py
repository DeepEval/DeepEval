import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, MaxPooling2D

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # 1x1 convolutional layer to reduce dimensionality
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    
    # Max pooling layer with stride of 2
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()