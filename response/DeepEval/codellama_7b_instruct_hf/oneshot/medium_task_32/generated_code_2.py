import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    input_groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Feature extraction for each group
    feature_groups = []
    for i in range(3):
        group = input_groups[i]
        conv1 = Conv2D(32, (1, 1), activation='relu')(group)
        conv2 = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')(conv1)
        conv3 = DepthwiseSeparableConv2D(32, (5, 5), activation='relu')(conv2)
        feature_groups.append(conv3)

    # Concatenate the feature groups and flatten
    concat = keras.layers.concatenate(feature_groups)
    flatten = Flatten()(concat)

    # Fully connected layer for classification
    dense = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model