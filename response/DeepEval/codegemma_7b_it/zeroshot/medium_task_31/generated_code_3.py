from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, concatenate, Dense, Flatten

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Split the image into three groups along the channel dimension
    group1 = Lambda(lambda x: tf.split(x, 3, axis=3)[0])(inputs)
    group2 = Lambda(lambda x: tf.split(x, 3, axis=3)[1])(inputs)
    group3 = Lambda(lambda x: tf.split(x, 3, axis=3)[2])(inputs)

    # Apply different convolutional kernels to each group
    conv_group1 = Conv2D(16, (1, 1), padding='same', activation='relu')(group1)
    conv_group2 = Conv2D(16, (3, 3), padding='same', activation='relu')(group2)
    conv_group3 = Conv2D(16, (5, 5), padding='same', activation='relu')(group3)

    # Concatenate the outputs from each group
    concat_features = concatenate([conv_group1, conv_group2, conv_group3])

    # Flatten the fused features
    flatten_features = Flatten()(concat_features)

    # Pass through two fully connected layers for classification
    dense_layer1 = Dense(128, activation='relu')(flatten_features)
    dense_layer2 = Dense(10, activation='softmax')(dense_layer1)

    # Create the model
    model = Model(inputs=inputs, outputs=dense_layer2)

    return model