import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)

    # Branch path
    branch_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Dense(units=128, activation='relu')(branch_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    
    # Reshape to match the input dimensions for multiplication
    branch_weights = keras.layers.Reshape((1, 1, 64))(branch_path)
    
    # Multiply the branch weights with the input
    weighted_input = Multiply()([input_layer, branch_weights])

    # Add the outputs from both paths
    combined_output = Add()([main_path, weighted_input])

    # Flatten and fully connected layers for final classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model