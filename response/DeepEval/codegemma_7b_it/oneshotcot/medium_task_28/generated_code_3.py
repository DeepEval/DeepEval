import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Lambda, add

def attention_block(input_tensor, filters):

    # Generate attention weights
    attention = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_tensor)
    attention = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(attention)
    attention = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(attention)

    # Multiply attention weights with input features
    contextualized_features = Multiply()([input_tensor, attention])

    return contextualized_features

def down_block(input_tensor, filters):

    # Reduce dimensionality and apply layer normalization and ReLU
    down_sampled = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding='valid')(input_tensor)
    down_sampled = BatchNormalization()(down_sampled)
    down_sampled = Activation('relu')(down_sampled)

    # Concatenate original and down-sampled features
    concat = Concatenate()([input_tensor, down_sampled])

    # Apply 3x3 convolution followed by ReLU activation
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # Apply 1x1 convolution to restore dimensionality
    conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv)
    conv = BatchNormalization()(conv)

    return conv

def up_block(input_tensor, skip_tensor, filters):

    # Upsample input tensor
    upsampled = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
    upsampled = BatchNormalization()(upsampled)
    upsampled = Activation('relu')(upsampled)

    # Concatenate with skip tensor
    concat = Concatenate()([upsampled, skip_tensor])

    # Apply 3x3 convolution followed by ReLU activation
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # Apply 1x1 convolution to restore dimensionality
    conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv)
    conv = BatchNormalization()(conv)

    return conv

def residual_block(input_tensor, filters):

    # Apply 3x3 convolution followed by ReLU activation
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # Apply another 3x3 convolution with ReLU activation
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # Add original input to the output
    residual = add([input_tensor, conv])

    return residual

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Attention block
    attention = attention_block(input_tensor, filters=64)

    # Down blocks
    down_1 = down_block(attention, filters=64)
    down_2 = down_block(down_1, filters=128)
    down_3 = down_block(down_2, filters=256)

    # Residual block
    residual = residual_block(down_3, filters=256)

    # Up blocks
    up_3 = up_block(residual, down_3, filters=128)
    up_2 = up_block(up_3, down_2, filters=64)
    up_1 = up_block(up_2, down_1, filters=32)

    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(up_1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model