import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda
import tensorflow as tf

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block: splitting the input into three groups
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Applying separable convolutions with different kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate outputs from the separable convolutions
    block1_output = Concatenate()([path1, path2, path3])

    # Second block: multiple branches for enhanced feature extraction
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1_output)

    # Branch 2: 1x1 Conv followed by two 3x3 Convolutions
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(block1_output)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)

    # Concatenate outputs from all branches
    block2_output = Concatenate()([branch1, branch2, branch3])

    # Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(block2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # To view the model architecture