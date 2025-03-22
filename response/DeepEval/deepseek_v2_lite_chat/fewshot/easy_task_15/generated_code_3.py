import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Local Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    dropout1 = Dropout(0.25)(avg_pool1)

    # Block 2: Additional Processing
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    dropout2 = Dropout(0.25)(avg_pool2)

    # Global Average Pooling
    avg_pool3 = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(dropout2)

    # Flatten and Fully Connected Layers
    flatten = Flatten()(avg_pool3)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()