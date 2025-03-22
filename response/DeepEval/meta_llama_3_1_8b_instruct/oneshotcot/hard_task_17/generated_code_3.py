import keras
from keras.layers import Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Multiply, Reshape, Add, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    block1_input = input_layer
    # Block 1: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(block1_input)
    # Reshape to original shape
    reshaped_global_avg_pool = Reshape((3, 3, 3))(global_avg_pool)

    # Weights are the outputs from the global average pooling layer
    weights = reshaped_global_avg_pool

    # Block 2: Deep Features Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_input)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Branch from Block 1
    branch = Multiply()([weights, max_pool])

    # Block 2: Main Path
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)

    # Fusion of the main path and the branch
    fused_output = Add()([max_pool2, branch])

    # Classification
    flat = Flatten()(fused_output)
    dense1 = Dense(units=64, activation='relu')(flat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model