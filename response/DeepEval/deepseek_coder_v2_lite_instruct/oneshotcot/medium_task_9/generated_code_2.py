import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add convolutional layer with 16 filters and kernel size 3x3
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Add batch normalization
    batch_norm1 = BatchNormalization()(conv1)

    # Step 4: Define a basic block
    def basic_block(input_tensor):
        # Path 1: Convolutional layer
        path1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Path 2: Batch normalization and ReLU activation
        path2 = BatchNormalization()(path1)
        path2 = ReLU()(path2)
        # Path 3: Convolutional layer
        path3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        # Path 4: Batch normalization
        path4 = BatchNormalization()(path3)
        # Add the outputs of both paths
        output_tensor = Add()([input_tensor, path4])
        output_tensor = ReLU()(output_tensor)
        return output_tensor

    # Step 5: Apply the basic block twice
    block1 = basic_block(batch_norm1)
    block2 = basic_block(block1)

    # Step 6: Add average pooling layer
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1)(block2)

    # Step 7: Flatten the output
    flatten_layer = Flatten()(avg_pool)

    # Step 8: Add dense layer for classification
    dense1 = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=dense1)

    return model