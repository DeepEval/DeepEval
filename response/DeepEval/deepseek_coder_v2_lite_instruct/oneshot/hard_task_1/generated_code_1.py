import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Parallel paths for feature extraction
    def block1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = GlobalAveragePooling2D()(path1)
        path1 = Dense(128, activation='relu')(path1)
        path1 = Dense(64, activation='relu')(path1)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = GlobalMaxPooling2D()(path2)
        path2 = Dense(128, activation='relu')(path2)
        path2 = Dense(64, activation='relu')(path2)

        output_tensor = Concatenate()([path1, path2])
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Activation('relu')(output_tensor)

        # Channel attention weights
        channel_weights = Dense(input_tensor.shape[-1], activation='sigmoid')(output_tensor)
        output_tensor = Multiply()([input_tensor, channel_weights])

        return output_tensor

    block1_output = block1(input_layer)

    # Block 2: Spatial feature extraction
    avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(block1_output)
    max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(block1_output)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    concat = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    concat = Activation('sigmoid')(concat)

    # Element-wise multiplication with Block 1 output
    block1_output = Multiply()([block1_output, concat])

    # Additional branch for output channel adjustment
    final_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    final_output = keras.layers.add([block1_output, final_branch])
    final_output = Activation('relu')(final_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(final_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])