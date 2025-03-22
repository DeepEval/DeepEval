import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Flatten, Add, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # First Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Skip connection
    skip_connection1 = Add()([input_layer, avg_pool1])

    # Second Block: Squeeze-and-Excitation (SE) block
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(skip_connection1)
    # Fully connected layers to generate channel weights
    se_dense1 = Dense(units=64, activation='relu')(gap)
    se_dense2 = Dense(units=skip_connection1.shape[-1], activation='sigmoid')(se_dense1)
    
    # Reshape to match the feature map dimensions
    scale = Reshape((1, 1, skip_connection1.shape[-1]))(se_dense2)
    # Scale the input features
    se_output = Multiply()([skip_connection1, scale])

    # Final fully connected layer for classification
    flatten = Flatten()(se_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model