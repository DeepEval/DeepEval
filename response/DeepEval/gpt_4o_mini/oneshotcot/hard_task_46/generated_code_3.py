import keras
import tensorflow as tf
from keras.layers import Input, SeparableConv2D, Concatenate, Lambda, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: Splitting input into three groups and applying separable convolutions
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    block1_output = Concatenate()([conv1, conv2, conv3])

    # Second block: Multiple branches for enhanced feature extraction
    branch1 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)

    branch2 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)

    block2_output = Concatenate()([branch1, branch2, branch3])

    # Final processing
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model