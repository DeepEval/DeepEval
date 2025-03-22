import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Batch normalization and ReLU activation
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        # 3x3 convolutional layer
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        return x

    # First pathway
    path1 = input_layer
    for _ in range(3):  # Repeat the block three times
        path1 = block(path1)
    # Concatenate the original input with the features from the block
    path1 = Concatenate()([input_layer, path1])
    
    # Second pathway
    path2 = input_layer
    for _ in range(3):  # Repeat the block three times
        path2 = block(path2)
    # Concatenate the original input with the features from the block
    path2 = Concatenate()([input_layer, path2])
    
    # Merge both pathways
    merged = Concatenate()([path1, path2])

    # Flatten the merged output
    flatten_layer = Flatten()(merged)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model