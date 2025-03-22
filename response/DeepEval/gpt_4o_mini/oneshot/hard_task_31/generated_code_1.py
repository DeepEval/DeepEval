import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First Block
    # Main Path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.5)(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch Path
    branch_path = input_layer

    # Adding both paths
    block_output = Add()([main_path, branch_path])

    # Second Block
    # Splitting the output into three groups using Lambda layer
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block_output)

    # Each group uses separable convolutional layers with different kernel sizes
    group_outputs = []
    for idx, kernel_size in enumerate([1, 3, 5]):
        x = Conv2D(filters=64, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_tensors[idx])
        x = Dropout(0.5)(x)
        group_outputs.append(x)

    # Concatenate the outputs from the three groups
    concatenated_output = Concatenate()(group_outputs)

    # Final layers for prediction
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model