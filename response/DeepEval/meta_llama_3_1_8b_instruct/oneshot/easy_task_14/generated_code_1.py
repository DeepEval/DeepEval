import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Define the input shape of the CIFAR-10 dataset

    # Compress the input features with global average pooling
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to generate weights
    weights = Dense(units=3, activation='relu')(global_avg_pooling)
    weights = Dense(units=3, activation='relu')(weights)

    # Reshape the weights to align with the input shape
    weights = Reshape((3, 1))(weights)

    # Multiply element-wise with the input feature map
    feature_map = Multiply()([input_layer, weights])

    # Flatten the result and pass it through another fully connected layer
    output_layer = Flatten()(feature_map)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model