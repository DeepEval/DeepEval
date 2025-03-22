import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolutional layer to reduce dimensionality to 16
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Basic block definition
    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        
        # Branch directly connected to input
        branch_path = input_tensor
        
        # Fuse the outputs of main path and branch path
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    # First basic block
    block1_output = basic_block(initial_conv)
    
    # Second basic block
    block2_output = basic_block(block1_output)

    # Additional convolutional layer in the branch
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(block1_output)

    # Fuse outputs of the second block and the additional branch
    combined_output = Add()([block2_output, branch_conv])

    # Average pooling layer to downsample feature maps
    avg_pooling = AveragePooling2D(pool_size=(2, 2))(combined_output)

    # Flatten the result
    flatten_layer = Flatten()(avg_pooling)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can create the model by calling dl_model()
model = dl_model()
model.summary()  # Print the model summary to confirm the architecture