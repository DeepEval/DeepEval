import keras
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Dense, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Block 2
    y = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    y = GlobalMaxPooling2D()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(10, activation='softmax')(y)

    # Channel attention
    channel_attention = Add()([x, y])
    channel_attention = Activation('relu')(channel_attention)
    channel_attention = Dense(32, activation='sigmoid')(channel_attention)
    channel_attention = Reshape((1, 1, 32))(channel_attention)

    # Spatial attention
    spatial_attention = Conv2D(16, (1, 1), activation='relu')(inputs)
    spatial_attention = GlobalMaxPooling2D()(spatial_attention)
    spatial_attention = Dense(128, activation='relu')(spatial_attention)
    spatial_attention = Dense(10, activation='sigmoid')(spatial_attention)
    spatial_attention = Reshape((1, 1, 16))(spatial_attention)

    # Feature fusion
    feature_fusion = Add()([channel_attention, spatial_attention])
    feature_fusion = Activation('relu')(feature_fusion)
    feature_fusion = Conv2D(32, (1, 1), activation='relu')(feature_fusion)

    # Final classification
    outputs = Dense(10, activation='softmax')(feature_fusion)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model