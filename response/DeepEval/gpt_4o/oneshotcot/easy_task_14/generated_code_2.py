import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten
from keras.models import Model

def dl_model():
    # Input layer with CIFAR-10 shape
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)

    # Step 2: Fully connected layers to generate weights for channels
    dense1 = Dense(units=32, activation='relu')(global_avg_pooling)  # Assuming 32 channels for simplicity
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Same as the number of input channels

    # Step 3: Reshape weights to match the input shape
    reshaped_weights = Reshape((1, 1, 3))(dense2)

    # Step 4: Multiply the weights element-wise with the input feature map
    scaled_features = Multiply()([input_layer, reshaped_weights])

    # Step 5: Flatten and final dense layer for classification
    flatten_layer = Flatten()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model