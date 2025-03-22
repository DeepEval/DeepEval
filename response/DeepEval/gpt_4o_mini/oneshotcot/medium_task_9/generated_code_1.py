import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    
    # Branch
    branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    
    # Feature fusion
    output_tensor = Add()([main_path, branch_path])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer to reduce dimensionality
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Two consecutive basic blocks
    block1_output = basic_block(conv1)
    block2_output = basic_block(block1_output)

    # Another convolutional layer for the branch
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)

    # Feature fusion again
    combined_output = Add()([block2_output, branch_conv])

    # Average pooling layer
    avg_pooling = AveragePooling2D(pool_size=(2, 2))(combined_output)
    
    # Flattening and fully connected layer
    flatten_layer = Flatten()(avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# To create the model, you can call
model = dl_model()