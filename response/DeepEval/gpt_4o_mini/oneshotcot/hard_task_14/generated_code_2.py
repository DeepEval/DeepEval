import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main Path
    main_path_global_pool = GlobalAveragePooling2D()(input_layer)
    main_path_fc1 = Dense(units=128, activation='relu')(main_path_global_pool)
    main_path_fc2 = Dense(units=3, activation='sigmoid')(main_path_fc1)  # Output size matches the number of channels

    # Reshape to match input layer's shape
    main_path_weights = Reshape((1, 1, 3))(main_path_fc2)  # Reshape to (1, 1, 3) for channel-wise multiplication

    # Element-wise multiplication with the original feature map
    main_path_output = keras.layers.multiply([input_layer, main_path_weights])  # Element-wise multiplication

    # Branch Path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)  # Match input channel size

    # Combine both paths
    combined_output = Add()([main_path_output, branch_path])  # Element-wise addition

    # Fully connected layers after combining paths
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model