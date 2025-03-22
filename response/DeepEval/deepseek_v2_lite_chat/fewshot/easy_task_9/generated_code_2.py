import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 1x1 convolutional layer to reduce dimensionality while maintaining stride
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from the 1x1 and 3x3 convolutional layers
    add_layer = Add()([conv1, depthwise_conv1])
    
    # Flattening layer
    flatten = Flatten()(add_layer)
    
    # Fully connected layer for classification
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer with 10 units for 10 classes (MNIST digits)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()