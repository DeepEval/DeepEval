import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Add, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Step 1: 1x1 Convolutional Layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: 3x3 Depthwise Separable Convolutional Layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # Step 3: 1x1 Convolutional Layer to reduce dimensionality
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Step 4: Add the output from conv2 to the original input
    added_output = Add()([conv2, input_layer])
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(added_output)
    
    # Step 6: Fully Connected Layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model