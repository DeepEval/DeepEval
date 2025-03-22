import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = Activation('relu')(batch_norm)

    # Compressing features using Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)
    
    # Fully connected layers to adjust dimensions
    fc1 = Dense(units=32, activation='relu')(global_avg_pool)
    fc2 = Dense(units=32, activation='sigmoid')(fc1)

    # Reshape to match the size of initial features
    reshaped_fc = keras.layers.Reshape((1, 1, 32))(fc2)

    # Multiply with initial features to generate weighted feature maps
    weighted_features = Multiply()([relu, reshaped_fc])

    # Concatenate with the input layer
    concat = Concatenate(axis=-1)([input_layer, weighted_features])

    # Dimensionality reduction and downsampling
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)

    # Final classification layer
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model