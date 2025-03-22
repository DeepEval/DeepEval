import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Input image shape is (32, 32, 3)

    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)

    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(conv3)  # 1x1 window
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)  # 2x2 window
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(conv3)  # 4x4 window

    # Flatten and concatenate
    concat = Concatenate()(pool1, pool2, pool3)
    flat = Flatten()(concat)
    concatenated = Concatenate()([flat, flat, flat])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])