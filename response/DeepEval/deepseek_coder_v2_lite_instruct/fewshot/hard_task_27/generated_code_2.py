import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Depthwise separable convolution layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(input_layer)
    bn = BatchNormalization()(depthwise_conv)
    relu = ReLU()(bn)

    # Flatten and fully connected layers
    flatten = Flatten()(relu)
    fc1 = Dense(units=32, activation='relu')(flatten)  # Assuming 32 channels as in the input
    fc2 = Dense(units=10, activation='softmax')(fc1)  # Output layer for 10 categories

    # Combining original input with processed features
    added = Add()([input_layer, fc2])

    model = Model(inputs=input_layer, outputs=added)
    
    return model

# Create the model
model = dl_model()
model.summary()