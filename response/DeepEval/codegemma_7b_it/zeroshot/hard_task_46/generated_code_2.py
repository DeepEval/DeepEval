from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, concatenate, Dense, GlobalAveragePooling2D, SeparableConv2D

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)

    # Feature extraction in first block
    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_inputs[0])
    tower_1 = SeparableConv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_inputs[1])
    tower_2 = SeparableConv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_inputs[2])
    tower_3 = SeparableConv2D(64, (7, 7), padding='same', activation='relu')(tower_3)

    # Concatenate outputs from first block
    concat_features = concatenate([tower_1, tower_2, tower_3])

    # Second block
    conv_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(concat_features)

    conv_2a = Conv2D(128, (1, 1), padding='same', activation='relu')(concat_features)
    conv_2a = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_2a)
    conv_2a = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_2a)

    pool = MaxPooling2D((3, 3), strides=(2, 2))(concat_features)

    # Concatenate outputs from second block
    concat_features = concatenate([conv_1, conv_2a, pool])

    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(concat_features)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(avg_pool)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)

    return model