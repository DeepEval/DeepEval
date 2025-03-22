import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(conv1x1)
    
    # 1x1 convolutional layer to reduce dimensionality
    conv1x1_reduced = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv3x3)
    
    # Adding the output from the 1x1 convolutional layer to the original input layer
    added = Add()([conv1x1_reduced, input_layer])
    
    # Flatten the result
    flatten_layer = Flatten()(added)
    
    # Fully connected layer to generate the final classification probabilities
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model