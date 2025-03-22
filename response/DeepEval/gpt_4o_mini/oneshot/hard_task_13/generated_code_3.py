import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block with four parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the outputs of the paths
    block1_output = Concatenate()([path1, path2, path3, path4])
    
    # Second block: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(block1_output)

    # Fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape to match the input of the feature map before multiplication
    reshape_layer = Reshape((1, 1, 32))(dense2)

    # Multiply the reshaped weights with the input from block 1
    feature_weighted = Multiply()([block1_output, reshape_layer])
    
    # Final fully connected layer to produce output probability distribution
    final_output = GlobalAveragePooling2D()(feature_weighted)  # Global Average Pooling again
    output_layer = Dense(units=10, activation='softmax')(final_output)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model