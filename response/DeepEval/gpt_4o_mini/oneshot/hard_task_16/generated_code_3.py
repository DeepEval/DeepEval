import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Concatenate, Lambda, GlobalMaxPooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    
    # Block 1
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[0])
        path1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path1)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path1)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[1])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path2)

        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[2])
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path3)

        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    global_pool = GlobalMaxPooling2D()(transition_conv)

    # Fully connected layers for weights
    weights = Dense(units=64, activation='relu')(global_pool)
    weights = Dense(units=transition_conv.shape[-1], activation='sigmoid')(weights)  # Output shape matches channels of transition_conv
    
    # Reshape weights to match adjusted output
    reshaped_weights = Reshape((1, 1, transition_conv.shape[-1]))(weights)
    
    # Main path output
    main_path_output = tf.multiply(transition_conv, reshaped_weights)

    # Branch directly connected to input
    branch_output = input_layer

    # Adding both paths
    added_output = Add()([main_path_output, branch_output])

    # Final classification layer
    final_output = Dense(units=10, activation='softmax')(added_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model