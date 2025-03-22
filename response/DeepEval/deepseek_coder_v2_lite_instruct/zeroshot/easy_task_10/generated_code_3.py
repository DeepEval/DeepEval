import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, MaxPooling2D

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
    
    # 1x1 convolutional layer to reduce dimensionality
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    
    # Apply stride of 2 to all convolutional layers
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()