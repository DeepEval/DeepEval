import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = Input(shape=(32, 32, 3))

    # Average Pooling Layers with different sizes
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each pooling layer's output
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Further flattening (although already flattened, we keep the structure)
    final_flatten = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(final_flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model