import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model