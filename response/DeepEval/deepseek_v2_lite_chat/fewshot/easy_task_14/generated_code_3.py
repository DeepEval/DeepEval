import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Assuming the CIFAR-10 input shape is 32x32x3

    # Compress the input features with global average pooling
    avg_pool = GlobalAveragePooling2D()(input_layer)

    # Fully connected layer to generate weights
    dense_layer1 = Dense(units=input_layer.shape[-1], activation='linear')(avg_pool)
    dense_layer2 = Dense(units=input_layer.shape[-1], activation='linear')(dense_layer1)

    # Reshape the weights to match the input shape
    dense_layer2_reshaped = Flatten()(dense_layer2)

    # Element-wise multiplication of the feature map with the learned weights
    output = keras.layers.multiply([input_layer, dense_layer2_reshaped])

    # Flatten the result and pass it through another fully connected layer
    flattened_output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model