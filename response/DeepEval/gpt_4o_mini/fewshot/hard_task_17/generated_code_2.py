import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Block 1: Global Average Pooling and Fully Connected Layers
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshaping the weights to match the input shape
    reshaped_weights = Reshape(target_shape=(1, 1, 32))(dense2)
    weighted_features = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication

    # Block 2: Convolutional Layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(weighted_features)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch from Block 1
    branch_output = Add()([max_pool, reshaped_weights])  # Adding the outputs from main path and branch

    # Final classification layers
    flatten = Flatten()(branch_output)
    final_dense1 = Dense(units=128, activation='relu')(flatten)
    final_dense2 = Dense(units=10, activation='softmax')(final_dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=final_dense2)

    return model