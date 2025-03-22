import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from keras.models import Model

def basic_block(input_tensor):
    # Main path
    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    # Branch path (identity mapping)
    branch = input_tensor

    # Adding the two paths
    output_tensor = Add()([relu, branch])

    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # First level - Basic block
    level1_output = basic_block(input_layer)

    # Second level - Two residual blocks
    level2_output = basic_block(level1_output)
    level2_output = basic_block(level2_output)

    # Third level - Capture features from initial convolution output
    conv_global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    level2_output = Add()([level2_output, conv_global_branch])

    # Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(level2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model