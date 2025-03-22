import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Main path
    main_path = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.2)(main_path)
    main_path = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)  # Restore channels

    # Branch path
    branch_path = input_layer

    # Adding both paths
    block_1_output = Add()([main_path, branch_path])

    # Block 2
    # Splitting the output into three groups
    split_output = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block_1_output)

    # Using Separable Convolution with different kernel sizes
    sep_conv1 = SeparableConv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(split_output[0])
    sep_conv1 = Dropout(0.2)(sep_conv1)

    sep_conv2 = SeparableConv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(split_output[1])
    sep_conv2 = Dropout(0.2)(sep_conv2)

    sep_conv3 = SeparableConv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(split_output[2])
    sep_conv3 = Dropout(0.2)(sep_conv3)

    # Concatenating the outputs from the three separable convolutions
    concatenated_output = Concatenate(axis=-1)([sep_conv1, sep_conv2, sep_conv3])

    # Final layers
    flatten_output = Flatten()(concatenated_output)
    output_layer = Dense(10, activation='softmax')(flatten_output)  # 10 classes for CIFAR-10

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to instantiate and compile the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()