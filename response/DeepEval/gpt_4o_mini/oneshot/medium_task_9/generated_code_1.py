import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    
    # Branch path
    branch_path = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)
    
    # Feature fusion
    output_tensor = Add()([main_path, branch_path])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    
    # Initial convolutional layer
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)
    
    # First basic block
    block1_output = basic_block(initial_conv)
    
    # Second basic block
    block2_output = basic_block(block1_output)
    
    # Final average pooling
    avg_pool = AveragePooling2D(pool_size=(8, 8))(block2_output)
    
    # Flattening the output
    flatten_layer = Flatten()(avg_pool)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model