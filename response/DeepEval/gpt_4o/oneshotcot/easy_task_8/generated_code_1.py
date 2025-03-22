import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    depthwise_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(rate=0.25)(depthwise_conv)
    
    # 1x1 convolutional layer
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv1x1)
    
    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(dropout2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model