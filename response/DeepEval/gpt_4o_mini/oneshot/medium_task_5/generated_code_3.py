import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Main Path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv1)

    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_pool1)
    main_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv2)

    # Branch Path
    branch_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv1)

    # Combine paths using addition
    combined_output = Add()([main_pool2, branch_pool1])

    # Flatten the output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model