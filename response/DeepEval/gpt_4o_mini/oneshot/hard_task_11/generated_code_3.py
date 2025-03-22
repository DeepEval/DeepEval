import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have dimensions 32x32 and 3 color channels

    # Main pathway with 1x1 convolution
    main_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel branch with 1x1, 1x3, and 3x1 convolutions
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs from the parallel branches
    concatenated_output = Concatenate()([path1, path2, path3])

    # Apply 1x1 convolution to the concatenated output
    combined_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_output)

    # Fusion with a direct connection from the input
    fused_output = Add()([main_path, combined_output])

    # Batch normalization
    batch_norm = BatchNormalization()(fused_output)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model