import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Main Path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    main_path = LayerNormalization()(main_path)

    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(main_path)

    # Branch Path
    branch_path = input_layer  # Directly connects to the input

    # Combine paths
    combined_output = Add()([main_path, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model