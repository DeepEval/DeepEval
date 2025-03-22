import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense, Concatenate, Lambda, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First Block
    def block_1(input_tensor):
        # Main Path
        conv_dropout = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.5)(conv_dropout)
        main_path_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)

        # Branch Path
        branch_path = input_tensor
        
        # Combine paths
        output_tensor = Add()([main_path_conv, branch_path])
        return output_tensor

    # Second Block
    def block_2(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Separable Convolutions with Dropout
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout2 = Dropout(rate=0.5)(conv1)

        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        dropout3 = Dropout(rate=0.5)(conv2)

        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        dropout4 = Dropout(rate=0.5)(conv3)

        # Concatenate outputs
        output_tensor = Concatenate()([dropout2, dropout3, dropout4])
        return output_tensor

    # Apply both blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Final processing for classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model