import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Deep feature extraction
    def path1_block(input_tensor):
        # First block of convolution
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        # Second block of convolution
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(batch_norm2)
        return avg_pool

    # Apply path1 to the input layer
    path1_output = path1_block(input_layer)

    # Path 2: Single convolutional layer
    conv_path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm_path2 = BatchNormalization()(conv_path2)
    avg_pool_path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(batch_norm_path2)

    # Concatenate outputs from both paths
    combined_features = Concatenate()([path1_output, avg_pool_path2])

    # Flatten the combined features
    flattened_features = Flatten()(combined_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()