import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Dense, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = GlobalAveragePooling2D()(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)  # Adjust units based on input shape and desired complexity
    main_weights = Reshape((1, 1, 32))(main_path)  # Reshape to match input shape
    main_output = Multiply()([input_layer, main_weights])  # Element-wise multiplication

    # Branch path (direct connection to input)
    branch_output = input_layer

    # Combine outputs from both paths
    combined_output = Add()([main_output, branch_output])

    # Additional fully connected layers
    fc1 = Dense(units=64, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model