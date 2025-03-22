import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate, Multiply, AveragePooling2D, Reshape
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)

    # Step 4: Define a block
    def block(input_tensor):
        # Step 4.1: Add convolutional layer as the first path
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.2: Add convolutional layer as the second path
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.3: Add convolutional layer as the third path
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.4: Add maxpooling layer as the fourth path
        path4 = GlobalAveragePooling2D()(input_tensor)
        path4 = Reshape((1, 1, 32))(path4)  # Adjust dimensions to match channels
        
        # Step 4.5: Add concatenate layer to merge the above paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor

    # Apply the block to the convolutional layer output
    block_output = block(conv)

    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(block_output)

    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)

    # Step 7: Add dense layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)

    # Step 8: Add dense layer
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Step 9: Add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model