import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    # Step 2: Add first convolutional layer
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    # Step 3: Add second convolutional layer
    main_conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(main_conv1)
    # Step 4: Add max pooling layer
    main_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(main_conv2)

    # Branch Path
    # Step 5: Add a single convolutional layer
    branch_conv = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)

    # Combine paths
    # Step 6: Concatenate the outputs of both paths
    combined = Concatenate()([main_pooling, branch_conv])
    
    # Step 7: Add flatten layer
    flatten_layer = Flatten()(combined)
    
    # Step 8: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    # Step 9: Add output dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model