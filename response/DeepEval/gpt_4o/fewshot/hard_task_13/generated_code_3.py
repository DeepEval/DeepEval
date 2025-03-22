import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First Block: Four parallel branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Second Block: Dimensionality reduction and feature scaling
    global_pooling = GlobalAveragePooling2D()(concatenated)
    
    # Two fully connected layers
    fc1 = Dense(units=128, activation='relu')(global_pooling)
    fc2 = Dense(units=concatenated.shape[-1], activation='sigmoid')(fc1)

    # Reshape fc2 to match the shape of concatenated for element-wise multiplication
    weights = Reshape(target_shape=(1, 1, concatenated.shape[-1]))(fc2)
    scaled_features = Multiply()([concatenated, weights])

    # Final classification layer
    output_layer = GlobalAveragePooling2D()(scaled_features)  # Pooling to reduce to single vector
    output_layer = Dense(units=10, activation='softmax')(output_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model