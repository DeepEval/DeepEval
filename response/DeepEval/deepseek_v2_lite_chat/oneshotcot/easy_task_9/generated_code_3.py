import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolution layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable 3x3 Convolution layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1)(conv1)
    depthwise_activation = keras.activations.relu(depthwise_conv)
    
    # 1x1 Convolution layer to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_activation)
    
    # Add the reduced dimensionality feature map to the original input
    added_feature = Add()[(1, 1)]([input_layer, conv2])
    
    # Max pooling with a stride of 2
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(added_feature)
    
    # Flatten layer
    flatten = Flatten()(max_pooling)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model