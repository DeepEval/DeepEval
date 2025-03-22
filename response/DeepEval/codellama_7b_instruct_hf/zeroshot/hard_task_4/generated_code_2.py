from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional block 1
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Channel attention block
    channel_attention = Dense(32, activation='relu')(x)
    channel_attention = Dense(32, activation='relu')(channel_attention)
    channel_attention = Dense(32, activation='sigmoid')(channel_attention)
    channel_attention = Reshape((32, 32))(channel_attention)
    x = Multiply()([x, channel_attention])

    # Fully connected block
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define model
    model = Model(inputs=input_layer, outputs=x)

    return model