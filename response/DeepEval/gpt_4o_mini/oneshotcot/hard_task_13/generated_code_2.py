import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First block with parallel branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate outputs
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Second block: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Generate weights for the input of block 2
    weights = Dense(units=concatenated.shape[-1], activation='sigmoid')(dense2)
    
    # Reshape weights to match the feature map shape
    reshaped_weights = Reshape((1, 1, concatenated.shape[-1]))(weights)

    # Element-wise multiplication with the concatenated feature map
    weighted_output = Multiply()([concatenated, reshaped_weights])

    # Final output layer
    final_output = Dense(units=10, activation='softmax')(weighted_output)

    # Construct the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model