import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 channels (RGB)

    # Average pooling layers with different pool sizes and strides
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs of the pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Further flatten the concatenated features
    final_flatten = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(final_flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model