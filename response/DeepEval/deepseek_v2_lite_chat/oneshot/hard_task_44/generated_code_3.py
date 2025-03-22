import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input shape for CIFAR-10
    input_shape = (32, 32, 3)
    input_layer = Input(shape=input_shape)

    # Block 1
    channel_splits = 3
    split = Lambda(lambda x: tf.split(x, channel_splits, axis=-1), name='channel_split')(input_layer)
    
    # Process each split separately
    def block_1(x):
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[0])  # 1x1 conv
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)  # 3x3 conv
        x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)  # 5x5 conv
        x = MaxPooling2D(pool_size=(1, 1), padding='same')(x)  # 1x1 max pooling
        return x

    def block_2(x):
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)  # 1x1 conv
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)  # 3x3 conv
        x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)  # 5x5 conv
        x = MaxPooling2D(pool_size=(1, 1), padding='same')(x)  # 1x1 max pooling
        return x

    # Apply dropout
    x = Lambda(lambda x: tf.keras.layers.Dropout(0.5)(x))(input_layer)

    # Concatenate outputs from the two blocks
    concat_output = Concatenate()(
        [block_1(split[0]), block_1(split[1]), block_1(split[2]), block_2(split)]
    )

    # Batch normalization and flatten
    bn_output = BatchNormalization()(concat_output)
    flattern_output = Flatten()(bn_output)

    # Fully connected layers
    fc_output = Dense(units=128, activation='relu')(flattern_output)
    output_layer = Dense(units=10, activation='softmax')(fc_output)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])