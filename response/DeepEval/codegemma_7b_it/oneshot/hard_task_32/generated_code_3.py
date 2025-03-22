import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def depthwise_separable_conv2d(input_tensor, filters, kernel_size, strides, padding):
    """Depthwise separable convolutional layer."""
    # Depthwise convolution
    conv_dw = Conv2D(filters, kernel_size, strides, padding, groups=filters)(input_tensor)
    # Batch normalization
    bn_dw = BatchNormalization()(conv_dw)

    # Pointwise convolution
    conv_pw = Conv2D(filters, (1, 1), strides, padding='same')(bn_dw)
    # Batch normalization
    bn_pw = BatchNormalization()(conv_pw)

    return bn_pw

def block(input_tensor):
    """Specialized block with depthwise separable convolution."""
    # Depthwise separable convolution
    conv_dw = depthwise_separable_conv2d(input_tensor, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    # Dropout
    drop_dw = Dropout(rate=0.25)(conv_dw)

    # 1x1 convolutional layer
    conv_pw = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(drop_dw)
    # Dropout
    drop_pw = Dropout(rate=0.25)(conv_pw)

    return drop_pw

def dl_model():
    """Deep learning model for image classification."""

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = block(input_tensor=input_layer)

    # Branch 2
    branch2 = block(input_tensor=input_layer)

    # Branch 3
    branch3 = block(input_tensor=input_layer)

    # Concatenate branch outputs
    concat = Concatenate()([branch1, branch2, branch3])

    # Flatten
    flatten = Flatten()(concat)

    # Fully connected layer 1
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Fully connected layer 2
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model