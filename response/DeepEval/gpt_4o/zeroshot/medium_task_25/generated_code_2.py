from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    num_classes = 10           # CIFAR-10 has 10 classes

    inputs = Input(shape=input_shape)

    # Path 1: Single 1x1 Convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Path 2: Average Pooling followed by 1x1 Convolution
    path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 Convolution followed by parallel 1x3 and 3x1 Convolutions
    path3 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    path3a = Conv2D(32, (1, 3), activation='relu', padding='same')(path3)
    path3b = Conv2D(32, (3, 1), activation='relu', padding='same')(path3)
    path3 = Concatenate()([path3a, path3b])

    # Path 4: 1x1 Convolution followed by 3x3 Convolution, then parallel 1x3 and 3x1 Convolutions
    path4 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    path4 = Conv2D(32, (3, 3), activation='relu', padding='same')(path4)
    path4a = Conv2D(32, (1, 3), activation='relu', padding='same')(path4)
    path4b = Conv2D(32, (3, 1), activation='relu', padding='same')(path4)
    path4 = Concatenate()([path4a, path4b])

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Classification layer
    flatten = Flatten()(concatenated)
    outputs = Dense(num_classes, activation='softmax')(flatten)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# To get a summary of the model architecture
model = dl_model()
model.summary()

# Optional: To visualize the model architecture
# plot_model(model, show_shapes=True, show_layer_names=True)