import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, Add, Reshape
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 channels (RGB)

    # Main path
    main_path_avg_pool = GlobalAveragePooling2D()(input_layer)
    main_path_fc1 = Dense(units=128, activation='relu')(main_path_avg_pool)
    main_path_fc2 = Dense(units=3, activation='sigmoid')(main_path_fc1)  # Output should match the channels of input

    # Reshape to match the input layer's shape
    main_path_weights = Reshape((1, 1, 3))(main_path_fc2)
    main_path_scaled = keras.layers.multiply([input_layer, main_path_weights])  # Element-wise multiplication

    # Branch path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined_output = Add()([main_path_scaled, branch_path])

    # Fully connected layers after combining the paths
    flatten_layer = GlobalAveragePooling2D()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model