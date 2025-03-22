import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model starts with a 1x1 convolutional layer to increase dimensionality, followed by a 3x3 depthwise separable convolutional layer for feature extraction. 
    Next, it applies another 1x1 convolutional layer to reduce dimensionality, maintaining a convolutional stride of 1 throughout. 
    The output from this layer is then added to the original input layer. 
    Finally, the processed output is passed through a flattening layer and a fully connected layer to generate the final classification probabilities.
    
    Parameters:
    None
    
    Returns:
    model: The constructed deep learning model.
    """
    
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 depthwise separable convolutional layer for feature extraction
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Step 4: Add 1x1 convolutional layer to reduce dimensionality
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Step 5: Add the output of the current layer to the original input layer
    output = Add()([conv3, input_layer])
    
    # Step 6: Apply batch normalization and flattening layer
    bath_norm = BatchNormalization()(output)
    flatten_layer = Flatten()(bath_norm)
    
    # Step 7: Apply fully connected layer to generate the final classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 8: Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model