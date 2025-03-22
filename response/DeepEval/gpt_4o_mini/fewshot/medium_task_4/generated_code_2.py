import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Path 1: Two blocks of convolution followed by average pooling
    def path_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pooling1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling1)
        conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        pooling2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
        
        return pooling2

    # Path 2: Single convolutional layer
    def path_2(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    # Get outputs from both paths
    path1_output = path_1(input_layer)
    path2_output = path_2(input_layer)

    # Combine both paths
    combined_output = Add()([path1_output, path2_output])

    # Flatten and classify
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model