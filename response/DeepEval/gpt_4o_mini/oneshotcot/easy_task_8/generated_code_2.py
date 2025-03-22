import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Add depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add dropout layer
    dropout1 = Dropout(0.5)(depthwise_conv)

    # Step 4: Add 1x1 convolutional layer for feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    
    # Step 5: Add dropout layer
    dropout2 = Dropout(0.5)(conv1x1)

    # Step 6: Flatten the result
    flatten_layer = Flatten()(dropout2)

    # Step 7: Add fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model