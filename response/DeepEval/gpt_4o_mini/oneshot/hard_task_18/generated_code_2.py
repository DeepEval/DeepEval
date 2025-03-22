import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First block: Feature extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Adding the input layer to the output of the first block
    main_path = Add()([input_layer, avg_pool])  # Element-wise addition

    # Second block: Channel weight generation and refinement
    global_avg_pool = GlobalAveragePooling2D()(main_path)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)  # Channel weights

    # Reshape the weights for multiplication
    reshape_weights = Reshape((1, 1, 32))(dense2)

    # Multiply the channel weights with the main path
    weighted_output = Multiply()([main_path, reshape_weights])

    # Flatten the output and pass through a final Dense layer for classification
    flatten_layer = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model