import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(batch_norm1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    batch_norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(batch_norm2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    batch_norm3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(batch_norm3)

    # Concatenate the outputs from all three blocks
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])

    # Flatten the concatenated output
    flat = Flatten()(concat)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flat)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])