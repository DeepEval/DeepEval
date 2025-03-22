import keras
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Multiply, Flatten

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Apply global average pooling to capture global information from the feature map
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Use two fully connected layers to generate weights whose size is the same as the channels of the input
    dense1 = Dense(units=3, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3, activation='relu')(dense1)

    # Reshape the generated weights to align with the input shape
    weights = Reshape(target_shape=input_shape[1:])(dense2)

    # Multiply the generated weights element-wise with the input feature map
    weighted_feature_map = Multiply()([input_layer, weights])

    # Flatten the weighted feature map
    flatten_layer = Flatten()(weighted_feature_map)

    # Apply a fully connected layer to obtain the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model