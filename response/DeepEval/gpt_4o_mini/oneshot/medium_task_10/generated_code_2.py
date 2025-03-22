import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def basic_block(input_tensor):
    # Main path
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Branch (shortcut connection)
    branch = input_tensor
    
    # Adding the main path and branch
    output_tensor = Add()([x, branch])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3
    x = basic_block(input_layer)  # First level

    # Second level with two residual blocks
    for _ in range(2):
        x = basic_block(x)

    # Main path for the third level
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Final output of the second level
    x = Add()([x, branch_conv])

    # Global branch convolution
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)

    # Final addition to enhance feature fusion
    final_output = Add()([x, global_branch])

    # Average Pooling and Fully Connected layer
    x = AveragePooling2D(pool_size=(8, 8))(final_output)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()