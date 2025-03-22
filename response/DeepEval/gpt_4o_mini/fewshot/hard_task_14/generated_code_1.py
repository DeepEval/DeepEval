import keras
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Conv2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)  # Global average pooling
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)  # First fully connected layer
    dense2 = Dense(units=3 * 32 * 32, activation='sigmoid')(dense1)  # Second fully connected layer to match input channels
    weights = Reshape((32, 32, 3))(dense2)  # Reshape to input layer's shape
    main_path_output = tf.multiply(input_layer, weights)  # Element-wise multiplication

    # Branch path
    branch_path_output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Final classification layers
    flatten_layer = Flatten()(combined_output)  # Flatten the combined output
    dense3 = Dense(units=256, activation='relu')(flatten_layer)  # Fully connected layer
    dense4 = Dense(units=128, activation='relu')(dense3)  # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense4)  # Output layer for 10 classes (CIFAR-10)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model