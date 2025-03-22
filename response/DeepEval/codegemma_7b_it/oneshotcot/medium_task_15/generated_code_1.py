import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Extract initial features with convolutional layer, batch normalization, and ReLU activation
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # Compress feature maps using global average pooling and two fully connected layers
    gap = GlobalAveragePooling2D()(act1)
    fc1 = Dense(units=100, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Reshape output to match initial feature size and generate weighted feature maps
    reshape_output = keras.layers.Reshape((32, 32, 64))(fc2)
    multiply_output = keras.layers.Multiply()([reshape_output, act1])

    # Concatenate weighted feature maps with input layer
    concat = keras.layers.Concatenate()([multiply_output, input_layer])
    
    # Reduce dimensionality and downsample feature using 1x1 convolution and average pooling
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    avg_pool = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Single fully connected layer for classification
    flatten = keras.layers.Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model