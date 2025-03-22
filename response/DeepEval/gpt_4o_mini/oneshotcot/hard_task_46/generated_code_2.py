import keras
import tensorflow as tf
from keras.layers import Input, SeparableConv2D, Concatenate, Lambda, MaxPooling2D, GlobalAveragePooling2D, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Split input into three groups and apply separable convolutions
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    block1_output = Concatenate()([path1, path2, path3])

    # Second Block: Multiple branches for enhanced feature extraction
    path4 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)

    path5 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    path5 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path5)
    path5 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path5)

    path6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1_output)

    block2_output = Concatenate()([path4, path5, path6])

    # Global Average Pooling and Fully Connected Layer
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model