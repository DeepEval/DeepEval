import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of (32, 32, 3)

    # Block 1
    def block1(input_tensor):
        x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)  # Split input into three groups
        features = []
        for i in range(3):
            conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[i])
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
            conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
            features.append(conv3)
        output_tensor = Concatenate()(features)
        return output_tensor

    block1_output = block1(input_tensor)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block2(input_tensor):
        conv = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        return conv

    block2_output = block2(transition_conv)

    # Main Path
    def main_path(input_tensor):
        shortcut = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        output_tensor = keras.layers.add([shortcut, conv])
        return output_tensor

    main_path_output = main_path(block2_output)

    # Branch
    def branch(input_tensor):
        conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        return conv

    branch_output = branch(input_tensor)

    # Final Output
    output_tensor = keras.layers.add([main_path_output, branch_output])
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    model = Model(inputs=input_layer, outputs=output_tensor)

    return model