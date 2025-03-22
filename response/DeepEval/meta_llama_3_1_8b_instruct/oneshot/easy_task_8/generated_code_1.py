import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    """
    This function creates a deep learning model for image classification using the MNIST dataset.
    The model begins with a depthwise separable convolutional layer to process the input data, 
    followed by a 1x1 convolutional layer for feature extraction. To mitigate overfitting, 
    a dropout layer is added after each convolutional layer. Finally, the output is flattened 
    and passed through a fully connected layer to generate classification probabilities.

    Args:
        None

    Returns:
        model (keras.Model): A compiled deep learning model for image classification.
    """

    input_layer = Input(shape=(28, 28, 1))
    # Depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Dropout layer to mitigate overfitting
    drop1 = Dropout(0.2)(depthwise_conv)
    # 1x1 convolutional layer for feature extraction
    conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(drop1)
    # Dropout layer to mitigate overfitting
    drop2 = Dropout(0.2)(conv)
    # Flatten the output
    flatten_layer = Flatten()(drop2)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model