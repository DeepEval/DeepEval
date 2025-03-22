import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Feature Extraction Paths
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    avg_pool = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    max_pool = GlobalMaxPooling2D()(conv1)
    avg_pool_max_pool = Concatenate()([avg_pool, max_pool])
    batch_norm = BatchNormalization()(avg_pool_max_pool)
    flatten = Flatten()(batch_norm)
    dense3 = Dense(units=128, activation='relu')(flatten)
    dense4 = Dense(units=64, activation='relu')(dense3)

    # Block 2: Spatial Feature Extraction
    avg_pool2 = GlobalAveragePooling2D()(conv1)
    max_pool2 = GlobalMaxPooling2D()(conv1)
    avg_pool_max_pool2 = Concatenate()([avg_pool2, max_pool2])
    avg_pool_spatial = Dense(units=512)(avg_pool2)
    max_pool_spatial = Dense(units=512)(max_pool2)
    avg_pool_spatial_dense = Dense(units=256)(avg_pool_spatial)
    max_pool_spatial_dense = Dense(units=256)(max_pool_spatial)

    # Combine Channels and Spatial Features
    concat = Concatenate()([dense4, avg_pool_spatial_dense, max_pool_spatial_dense])
    conv2 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='sigmoid')(concat)
    output = Activation('sigmoid')(conv2)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])