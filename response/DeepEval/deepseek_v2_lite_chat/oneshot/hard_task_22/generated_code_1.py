import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel
    channel_split1 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    channel_split2 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    channel_split3 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Multi-scale feature extraction with separable convolutional layers
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(channel_split1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(channel_split2)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(channel_split3)

    # Concatenate the outputs of the three groups
    concat_layer = Concatenate()(
        [conv1, conv2, conv3]
    )

    # 1x1 convolutional layer to align the number of output channels with those of the main path
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat_layer)

    # Add the outputs of the main path and the branch path
    fused_output = tf.keras.layers.Add()([concat_layer, conv4])

    # Flatten the result and pass through two fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model