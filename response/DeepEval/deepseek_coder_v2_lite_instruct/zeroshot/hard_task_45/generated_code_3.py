import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First block: Split into three groups and apply depthwise separable convolutions
    split_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    def depthwise_separable_conv(x, kernel_size):
        depthwise_conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_initializer='he_normal', depthwise_constraint=None, pointwise_initializer='he_normal', pointwise_constraint=None)(x)
        pointwise_conv = Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        return pointwise_conv
    
    conv1x1 = depthwise_separable_conv(split_1[0], kernel_size=(1, 1))
    conv3x3 = depthwise_separable_conv(split_1[1], kernel_size=(3, 3))
    conv5x5 = depthwise_separable_conv(split_1[2], kernel_size=(5, 5))
    
    concatenated_output_1 = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])
    
    # Second block: Multiple branches for feature extraction
    def branch_extraction(x, config):
        outputs = []
        for layer in config:
            if layer == '1x1':
                x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
            elif layer == '<1x1':
                x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
            elif layer == '3x3':
                x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        outputs.append(x)
        return outputs
    
    branch1 = branch_extraction(concatenated_output_1, ['1x1', '3x3'])
    branch2 = branch_extraction(concatenated_output_1, ['<1x1', '3x3'])
    branch3 = branch_extraction(concatenated_output_1, ['maxpool', '1x1'])
    
    concatenated_output_2 = Concatenate(axis=-1)(branch1 + branch2 + branch3)
    
    # Flatten and fully connected layer
    x = Flatten()(concatenated_output_2)
    outputs = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()