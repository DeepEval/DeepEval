import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # First pathway
    def block(input_tensor):
        x = BatchNormalization()(input_tensor)
        x = ReLU()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = Concatenate()([input_tensor, x])
        return x

    # Repeated block structure for the first pathway
    path1 = input_layer
    for _ in range(3):
        path1 = block(path1)

    # Second pathway (same structure)
    path2 = input_layer
    for _ in range(3):
        path2 = block(path2)

    # Concatenate outputs from both pathways
    merged_output = Concatenate()([path1, path2])

    # Flatten the merged output
    flatten_layer = Flatten()(merged_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model