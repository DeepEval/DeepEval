import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def path_1(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        return conv1x1

    def path_2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)
        return conv1x1

    def path_3(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(conv1x1)
        conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(conv1x1)
        concat = Concatenate()([conv1x3, conv3x1])
        return concat

    def path_4(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(conv3x3)
        conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(conv3x3)
        concat = Concatenate()([conv1x3, conv3x1])
        return concat

    # Apply paths to input
    p1 = path_1(input_layer)
    p2 = path_2(input_layer)
    p3 = path_3(input_layer)
    p4 = path_4(input_layer)

    # Concatenate outputs of all paths
    merged = Concatenate()([p1, p2, p3, p4])

    # Flatten the concatenated output
    flattened = Flatten()(merged)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Define and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model