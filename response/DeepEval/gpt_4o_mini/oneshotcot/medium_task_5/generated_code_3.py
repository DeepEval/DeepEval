import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    # First block of convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second block of convolution
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(max_pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch Path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_conv = MaxPooling2D(pool_size=(2, 2))(branch_conv)

    # Combine the main path and the branch path
    combined = Add()([max_pool2, branch_conv])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model