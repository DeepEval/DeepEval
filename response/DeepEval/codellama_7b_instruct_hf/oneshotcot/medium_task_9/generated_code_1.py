import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Flatten, Dense

def dl_model():

    # Define the input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with 16 filters and kernel size (3, 3)
    conv1 = Conv2D(16, (3, 3), activation='relu')(input_layer)

    # Batch normalization
    batch_norm1 = BatchNormalization()(conv1)

    # ReLU activation
    relu1 = ReLU()(batch_norm1)

    # Convolutional layer with 16 filters and kernel size (3, 3)
    conv2 = Conv2D(16, (3, 3), activation='relu')(relu1)

    # Batch normalization
    batch_norm2 = BatchNormalization()(conv2)

    # ReLU activation
    relu2 = ReLU()(batch_norm2)

    # Average pooling layer with pool size (2, 2)
    avg_pool = GlobalAveragePooling2D()(relu2)

    # Flatten the output
    flat = Flatten()(avg_pool)

    # Fully connected layer with 128 units and ReLU activation
    dense1 = Dense(128, activation='relu')(flat)

    # Fully connected layer with 10 units and softmax activation
    output = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model