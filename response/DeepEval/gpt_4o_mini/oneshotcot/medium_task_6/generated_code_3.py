import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)
    
    # Step 2: Add initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Define parallel blocks
    def parallel_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        activated = keras.activations.relu(norm)
        return activated

    # Step 4: Create three parallel blocks
    block1 = parallel_block(initial_conv)
    block2 = parallel_block(initial_conv)
    block3 = parallel_block(initial_conv)

    # Step 5: Add the outputs of the blocks to the initial convolution's output
    added_output = Add()([initial_conv, block1, block2, block3])

    # Step 6: Flatten the result
    flatten_layer = Flatten()(added_output)

    # Step 7: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 8: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model