import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, SeparableConv2D

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 depthwise separable convolutional layer
    sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Add another 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(sep_conv)
    
    # Step 5: Add layer to merge processed output with the original input layer
    add_layer = Add()([conv2, input_layer])
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(add_layer)
    
    # Step 7: Add fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 8: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model