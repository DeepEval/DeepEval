import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path: Two blocks of Conv + Max Pooling
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)

    # Branch path: One block of Conv + Max Pooling
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path)

    # Combine main path and branch path using addition
    combined = Add()([main_path, branch_path])

    # Flatten the output
    flatten_layer = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model