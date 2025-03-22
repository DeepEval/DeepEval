import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Main path: two consecutive 3x3 convolutional layers followed by max pooling
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_path_conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(main_path_conv1)
    main_path_pool = MaxPooling2D(pool_size=(2, 2))(main_path_conv2)

    # Branch path: single 5x5 convolutional layer
    branch_path_conv = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)

    # Concatenate both paths
    combined_features = Concatenate()([main_path_pool, branch_path_conv])

    # Flatten the concatenated features
    flatten_layer = Flatten()(combined_features)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model