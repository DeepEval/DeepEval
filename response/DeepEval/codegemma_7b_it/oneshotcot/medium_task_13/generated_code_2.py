import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)

    # Second convolutional layer
    concat_input = Concatenate()([inputs, conv1])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_input)

    # Third convolutional layer
    concat_input = Concatenate()([inputs, conv1, conv2])
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_input)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Batch normalization layer
    batch_norm = BatchNormalization()(max_pooling)

    # Flatten layer
    flatten_layer = Flatten()(batch_norm)

    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Second fully connected layer
    outputs = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])