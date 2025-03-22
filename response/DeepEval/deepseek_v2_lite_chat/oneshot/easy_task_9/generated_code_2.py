import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolution Layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Depthwise Separable Convolution Layer
    dwconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    dwconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(dwconv)
    
    # 1x1 Convolution Layer for Dimension Reduction
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(dwconv)

    # Adding the output of conv2 to the original input
    add_layer = Add()([conv1, conv2])
    
    # Max Pooling Layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(add_layer)

    # Flattening Layer
    flatten_layer = Flatten()(max_pooling)

    # Fully Connected Layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model