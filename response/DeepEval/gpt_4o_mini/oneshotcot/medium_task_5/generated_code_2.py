import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, BatchNormalization

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Main Path
    # Step 2: First block of convolution
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    max_pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_main)

    # Step 3: Second block of convolution
    conv3_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_main)
    conv4_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_main)
    max_pool_main_final = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4_main)

    # Branch Path
    # Step 4: Single block of convolution
    conv1_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_branch)

    # Step 5: Combine main and branch paths
    combined = Add()([max_pool_main_final, max_pool_branch])

    # Step 6: Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Step 7: Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 8: Output layer with 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model