import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Pathway 1
    conv1_path1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2_path1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_path1)
    avg_pool_path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_path1)

    # Pathway 2
    conv_path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine the outputs of both pathways
    combined = Add()([avg_pool_path1, conv_path2])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model