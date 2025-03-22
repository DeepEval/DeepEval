import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main Path
    main_path_initial = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path_initial)

    # Branch 2
    branch2 = MaxPooling2D(pool_size=(2, 2))(main_path_initial)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3
    branch3 = MaxPooling2D(pool_size=(2, 2))(main_path_initial)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Final convolution in the main path
    main_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch Path
    branch_path_initial = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the outputs of both paths
    merged_output = Add()([main_path_output, branch_path_initial])

    # Fully connected layers for classification
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model