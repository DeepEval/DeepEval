import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block: average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(avg_pool1)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(avg_pool2)

    # Flatten and concatenate the outputs from pooling layers
    flat_concat = Flatten()(Concatenate()([avg_pool3, avg_pool2, avg_pool1]))

    # Second block: multi-scale feature extraction
    def multi_scale_features(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(path2)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path3)
        dropout1 = Dropout(0.5)(path4)
        return dropout1

    path1_output = multi_scale_features(flat_concat)
    path2_output = multi_scale_features(path1_output)
    path3_output = multi_scale_features(path2_output)
    path4_output = multi_scale_features(path3_output)

    # Concatenate the outputs from all paths
    concat_output = Concatenate(axis=-1)([path1_output, path2_output, path3_output, path4_output])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])