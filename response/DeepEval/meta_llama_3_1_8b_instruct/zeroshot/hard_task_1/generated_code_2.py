from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, concatenate, AveragePooling2D, MaxPooling2D, Conv2DTranspose, multiply, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Block 1: Channel Attention
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(32, activation='relu')(path1)
    path1 = Dense(32)(path1)

    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(32, activation='relu')(path2)
    path2 = Dense(32)(path2)

    channel_attention = add([path1, path2])
    channel_attention = Activation('sigmoid')(channel_attention)
    x = multiply([x, channel_attention])

    # Block 2: Spatial Attention
    avg_pool = AveragePooling2D()(x)
    max_pool = MaxPooling2D()(x)
    spatial_attention = concatenate([avg_pool, max_pool])
    spatial_attention = Conv2D(32, (1, 1), activation='relu')(spatial_attention)
    spatial_attention = Activation('sigmoid')(spatial_attention)
    x = multiply([x, spatial_attention])

    # Align Output Channels
    x = Conv2D(32, (1, 1), activation='relu')(x)

    # Final Classification
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    model = Model(inputs, x)
    return model

# Test the model
model = dl_model()
model.summary()