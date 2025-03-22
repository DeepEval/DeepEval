import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Reshape, Permute, DepthwiseConv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_inputs[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_inputs[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_inputs[2])
    
    block1_output = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Block 2
    shape = tf.shape(block1_output)
    reshaped = Reshape((shape[1], shape[2], 3, shape[3] // 3))(block1_output)
    permuted = Permute((0, 1, 3, 2))(reshaped)
    block2_output = Reshape(shape[1:])(permuted)

    # Block 3
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(block2_output)

    # Branch directly from input
    branch_output = input_layer

    # Combine main path and branch
    combined_output = Add()([depthwise_conv, branch_output])

    # Fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=dense_layer)

    return model

# Create the model
model = dl_model()
model.summary()  # Optional: To display the model architecture