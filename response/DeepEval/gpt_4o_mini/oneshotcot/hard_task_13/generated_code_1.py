import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Block 1: Parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)

    # Concatenate outputs of the four branches
    block_output = Concatenate()([path1, path2, path3, path4])

    # Block 2: Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(block_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Generating weights
    weight_shape = (1, 1, 64)  # Reshape to match the number of channels in the previous layer
    weights = Dense(units=64, activation='sigmoid')(dense2)
    reshaped_weights = Reshape(weight_shape)(weights)

    # Element-wise multiplication with the feature map
    multiplied_output = Multiply()([block_output, reshaped_weights])

    # Final fully connected layer for output
    final_output = Dense(units=10, activation='softmax')(multiplied_output)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model

# You can create the model by calling the dl_model function
model = dl_model()
model.summary()  # Display the model summary