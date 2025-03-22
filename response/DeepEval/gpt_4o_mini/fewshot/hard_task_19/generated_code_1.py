import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch path
    branch_path = GlobalAveragePooling2D()(input_layer)
    fc1 = Dense(units=128, activation='relu')(branch_path)
    fc2 = Dense(units=32, activation='sigmoid')(fc1)  # Channel weights
    reshaped_weights = Reshape((1, 1, 32))(fc2)  # Reshape to match input dimensions

    # Apply the weights to the input
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Combine paths
    combined_output = Add()([main_path, weighted_input])

    # Final classification layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model