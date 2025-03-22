import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Lambda
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    conv2 = SeparableConv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)

    # Compute channel attention weights through global average pooling
    avg_pool = GlobalAveragePooling2D()(conv2)
    channel_attention = Dense(units=32)(avg_pool)
    channel_attention = Activation('relu')(channel_attention)
    channel_attention = Dense(units=32)(channel_attention)
    channel_attention = Activation('softmax')(channel_attention)

    # Generate weights whose size is the same as the channels of the initial features
    attention_weighted_features = Lambda(lambda x: K.dot(x[0], x[1]))([conv2, channel_attention])

    # Reshape the attention weights to match the initial features and multiply with the initial features
    attention_weighted_features = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_weighted_features)

    # Combine the attention-weighted features with the initial input
    combined_features = Concatenate()([conv2, attention_weighted_features])

    # Reduce the dimensionality with a 1x1 convolution
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(combined_features)

    # Flattening layer
    flatten_layer = Flatten()(conv3)

    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model