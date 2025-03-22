import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, UpSampling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch for local feature extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # Pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(conv2)

    # Second and third branches
    # Average pooling
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    # Upsampling
    up_conv1 = UpSampling2D(size=(2, 2))(avg_pool1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_conv1)
    # Pooling
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Concatenation
    concat = Concatenate(axis=-1)([conv2, conv3, avg_pool, avg_pool2])

    # 1x1 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Flatten and fully connected layers
    flatten = Flatten()(conv4)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])