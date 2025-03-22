import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(x):
        # Depthwise separable convolutions
        path1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        path1 = Conv2D(32, (3, 3), padding='same', activation='relu', depthwise_constraint=None)(path1)
        path2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        path2 = Conv2D(32, (5, 5), padding='same', activation='relu', depthwise_constraint=None)(path2)
        path3 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        # Concatenate outputs
        return Concatenate()([path1, path2, path3])

    block1_output = first_block(input_layer)
    block1_output = BatchNormalization()(block1_output)

    # Second block
    def second_block(x):
        # Multiple branches
        branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
        branch3 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
        branch4 = MaxPooling2D((3, 3), strides=1, padding='same')(x)
        branch4 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4)
        # Concatenate outputs
        return Concatenate()([branch1, branch2, branch3, branch4])

    block2_output = second_block(block1_output)
    block2_output = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()