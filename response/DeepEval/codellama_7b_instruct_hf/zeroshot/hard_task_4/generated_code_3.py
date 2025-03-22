import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Conv2D,
    BatchNormalization,
    Add,
    Activation
)
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Add a 1x1 convolution to increase the dimensionality of the input channels
    x = Conv2D(filters=3 * 3, kernel_size=1, strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = Conv2D(filters=3 * 3, kernel_size=3, strides=1, padding='same', use_bias=False, depthwise_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Compute channel attention weights using global average pooling
    channel_attention_weights = GlobalAveragePooling2D()(x)

    # Add two fully connected layers to generate weights with same size as the channels of the initial features
    channel_attention_weights = Dense(units=3 * 3)(channel_attention_weights)
    channel_attention_weights = BatchNormalization()(channel_attention_weights)
    channel_attention_weights = Activation('relu')(channel_attention_weights)
    channel_attention_weights = Dense(units=3 * 3)(channel_attention_weights)
    channel_attention_weights = BatchNormalization()(channel_attention_weights)
    channel_attention_weights = Activation('sigmoid')(channel_attention_weights)

    # Reshape the attention weights to match the initial features
    channel_attention_weights = Reshape((3, 3))(channel_attention_weights)

    # Multiply the initial features with the attention weights
    x = Add()([x, channel_attention_weights])

    # Reduce the dimensionality of the output using a 1x1 convolution
    x = Conv2D(filters=3 * 3, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Flatten the output and add a fully connected layer for classification
    x = Flatten()(x)
    x = Dense(units=10)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(units=10)(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model