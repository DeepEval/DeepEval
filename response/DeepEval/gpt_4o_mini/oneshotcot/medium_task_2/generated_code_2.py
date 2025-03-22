import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Define the main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Step 3: Define the branch path
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)

    # Step 4: Concatenate both paths
    combined = Concatenate()([main_path, branch_path])

    # Step 5: Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Step 6: Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 7: Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model