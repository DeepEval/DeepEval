import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1_conv)

    # Block 2
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_pooling)
    block2_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2_conv)

    # Concatenate the outputs from both paths
    concat = Add()([block1_pooling, block2_pooling])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])