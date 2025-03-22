import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Branch
    branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)
    
    # Feature fusion
    output_tensor = Add()([relu, branch])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)
    
    # First basic block
    block1_output = basic_block(initial_conv)
    
    # Second basic block
    block2_output = basic_block(block1_output)
    
    # Final output after average pooling
    avg_pool = AveragePooling2D(pool_size=(8, 8))(block2_output)  # Downsampling to 4x4 feature maps
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # This line will print the model summary