import keras
from keras import initializers
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Lambda, Add
from keras.regularizers import l2

def depthwise_conv2d_bn(inputs, kernel_size, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4)):
    """
    Performs depthwise separable convolution followed by batch normalization.
    """
    conv = Conv2D(kernel_size=kernel_size, strides=strides, padding=padding, depthwise_initializer='he_normal', 
                 kernel_regularizer=kernel_regularizer)(inputs)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    return act

def pointwise_conv2d_bn(inputs, filters, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4)):
    """
    Performs pointwise convolution followed by batch normalization.
    """
    conv = Conv2D(filters=filters, strides=strides, padding=padding, kernel_initializer='he_normal',
                 kernel_regularizer=kernel_regularizer)(inputs)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    return act

def spatial_features(inputs):
    """
    Extracts spatial features using a 7x7 depthwise separable convolutional layer.
    """
    x = depthwise_conv2d_bn(inputs, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = pointwise_conv2d_bn(x, filters=64, strides=(1, 1), padding='same')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def channel_wise_features(inputs):
    """
    Performs channel-wise feature transformation using two fully connected layers.
    """
    x = Flatten()(inputs)
    x = Dense(units=64, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dense(units=64, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    return x

def combine_features(inputs1, inputs2):
    """
    Combines original input with processed features through an addition operation.
    """
    return Add()([inputs1, inputs2])

def output_layer(inputs):
    """
    Classifies images into 10 categories using final two fully connected layers.
    """
    x = Dense(units=10, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    return x

def dl_model():
    """
    Constructs the complete model architecture.
    """
    inputs = Input(shape=(32, 32, 3))
    spatial_features_output = spatial_features(inputs)
    channel_wise_features_output = channel_wise_features(inputs)
    combined_features = combine_features(spatial_features_output, channel_wise_features_output)
    output = output_layer(combined_features)
    model = keras.Model(inputs=inputs, outputs=output)
    return model