import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main path
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Split into three branches
    # First branch with 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Second branch: average pooling followed by 3x3 convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    # Third branch: average pooling followed by 3x3 convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)

    # Concatenate the outputs of the branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution after concatenation
    main_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_output)

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusing the main and branch paths
    fused_output = Add()([main_output, branch_path])
    
    # Batch normalization
    fused_output = BatchNormalization()(fused_output)

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model