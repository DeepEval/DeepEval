import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Global Average Pooling to capture global information
    gap_layer = GlobalAveragePooling2D()(input_layer)

    # Step 3: Two fully connected layers to generate weights
    fc1 = Dense(units=512, activation='relu')(gap_layer)  # First fully connected layer
    fc2 = Dense(units=512, activation='relu')(fc1)         # Second fully connected layer

    # Step 4: Reshape the weights to align with the input shape (32x32x3)
    reshaped_weights = Reshape((1, 1, 512))(fc2)  # Reshape to (1, 1, 512) for multiplication

    # Step 5: Multiply element-wise with the input feature map
    multiplied = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication

    # Step 6: Flatten the result
    flatten_layer = Flatten()(multiplied)

    # Step 7: Add another fully connected layer to produce final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model