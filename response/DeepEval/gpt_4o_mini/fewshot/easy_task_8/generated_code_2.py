import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolution layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Dropout to mitigate overfitting
    dropout1 = Dropout(0.25)(depthwise_conv)

    # 1x1 convolutional layer for feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    # Dropout to mitigate overfitting
    dropout2 = Dropout(0.25)(conv1x1)

    # Flatten the output
    flatten_layer = Flatten()(dropout2)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model