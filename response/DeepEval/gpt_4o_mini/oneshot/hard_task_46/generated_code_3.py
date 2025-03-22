import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: Splitting input into three groups and applying separable convolutions
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply separable convolutions with different kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    
    # Concatenate the outputs of the three paths
    block1_output = Concatenate()([path1, path2, path3])
    
    # Second block: Multiple branches for enhanced feature extraction
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)

    # 1x1 convolution followed by two 3x3 convolutions
    series_branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    series_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(series_branch)
    series_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(series_branch)

    # Max pooling branch
    max_pool_branch = MaxPooling2D(pool_size=(2, 2))(block1_output)

    # Concatenate all branches
    block2_output = Concatenate()([conv_branch, series_branch, max_pool_branch])

    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(block2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model