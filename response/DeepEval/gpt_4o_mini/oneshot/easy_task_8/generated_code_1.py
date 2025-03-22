import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Dropout to mitigate overfitting
    dropout1 = Dropout(0.25)(depthwise_conv)
    
    # 1x1 convolutional layer for feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
    # Dropout after the convolutional layer
    dropout2 = Dropout(0.25)(conv1x1)
    
    # Flatten the output
    flatten_layer = Flatten()(dropout2)
    
    # Fully connected layer for classification
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_output)

    return model

# Example of creating the model
model = dl_model()
model.summary()