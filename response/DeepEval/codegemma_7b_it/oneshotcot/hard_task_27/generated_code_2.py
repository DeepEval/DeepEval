import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, concatenate, Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Extract spatial features with a 7x7 depthwise separable convolutional layer
    conv_1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', use_bias=False)(input_layer)
    norm_1 = BatchNormalization()(conv_1)
    act_1 = keras.activations.relu(norm_1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(act_1)
    
    # Two fully connected layers with the same numbers of channel as the input layer for channel-wise feature transformation
    conv_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(max_pool_1)
    norm_2 = BatchNormalization()(conv_2)
    act_2 = keras.activations.relu(norm_2)
    conv_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(max_pool_1)
    norm_3 = BatchNormalization()(conv_3)
    act_3 = keras.activations.relu(norm_3)
    
    # Combine the original input with the processed features through an addition operation
    concat_features = concatenate([act_1, act_2, act_3], axis=-1)
    
    # Final two fully connected layers for classification
    conv_4 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(concat_features)
    norm_4 = BatchNormalization()(conv_4)
    act_4 = keras.activations.relu(norm_4)
    flatten_layer = Flatten()(act_4)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model