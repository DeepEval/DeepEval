import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Path 1: Main Path
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    # Add the paths
    branch_path = input_layer
    # Combine the paths using an Add operation
    combined_tensor = Add()([conv2, branch_path])
    # Batch Normalization and Flatten layers
    batch_norm = BatchNormalization()(combined_tensor)
    flatten_layer = Flatten()(batch_norm)
    # Dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()
model.summary()