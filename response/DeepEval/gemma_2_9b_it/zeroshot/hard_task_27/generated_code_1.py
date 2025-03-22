import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    
    input_tensor = Input(shape=(32, 32, 3))

    # Depthwise separable convolution with layer normalization
    x = Conv2D(32, kernel_size=(7, 7), depth_multiplier=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Two fully connected layers for channel-wise feature transformation
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Combine original input with processed features
    x = Add()([input_tensor, x])

    # Final two fully connected layers for classification
    x = Flatten()(x)
    output_tensor = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model