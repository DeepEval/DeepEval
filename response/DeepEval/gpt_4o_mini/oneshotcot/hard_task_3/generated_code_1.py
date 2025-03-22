import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Flatten, Dense, Lambda

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension into three groups
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def feature_extraction_branch(tensor):
        # Each branch applies a 1x1 convolution followed by a 3x3 convolution
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return x

    # Process each split tensor through the feature extraction branch
    branch_outputs = [feature_extraction_branch(tensor) for tensor in split_tensors]

    # Concatenate outputs from the three branches
    main_path = Concatenate()(branch_outputs)

    # Dropout layer for feature selection to mitigate overfitting
    main_path = Dropout(rate=0.5)(main_path)

    # Branch pathway processing through a 1x1 convolution to match dimensions
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both pathways using addition
    combined = Add()([main_path, branch_path])

    # Flatten the combined output before the final classification layer
    flatten_layer = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model