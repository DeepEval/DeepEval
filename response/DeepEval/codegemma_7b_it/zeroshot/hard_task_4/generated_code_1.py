from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, AveragePooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Dense, Conv2DTranspose, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Increase dimensionality of input channels threefold
    x = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(inputs)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Compute channel attention weights
    x_global = GlobalAveragePooling2D()(x)
    x_fc1 = Dense(units=x.shape[3]//2, activation='relu')(x_global)
    x_fc2 = Dense(units=x.shape[3], activation='sigmoid')(x_fc1)

    # Reshape weights to match initial features
    attention_weights = Reshape((1, 1, x.shape[3]))(x_fc2)

    # Channel attention weighting
    x = multiply([x, attention_weights])

    # Reduce dimensionality and combine with initial input
    x = Conv2DTranspose(filters=3, kernel_size=(1, 1), padding='same')(x)
    outputs = add([x, inputs])

    # Classification process
    outputs = Flatten()(outputs)
    outputs = Dense(units=10, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model