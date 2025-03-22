import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting and applying depthwise separable convolutions
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Convolutional branches with different kernel sizes
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split[2])

    # Concatenate outputs of first block
    block1_output = Concatenate()([conv1, conv2, conv3])

    # Block 2: Multiple branches for feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(block1_output)
    
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(block1_output)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(block1_output)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(block1_output)
    branch4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch4)
    
    branch5 = MaxPooling2D(pool_size=(2, 2))(block1_output)
    branch5 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch5)

    # Concatenate outputs of second block
    block2_output = Concatenate()([branch1, branch2, branch3, branch4, branch5])

    # Flatten and fully connected layers for classification
    flatten = Flatten()(block2_output)
    output = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # This will print the model summary