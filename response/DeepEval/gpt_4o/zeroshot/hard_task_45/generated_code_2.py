import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image size
    inputs = Input(shape=input_shape)

    # First Block
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    group_1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layer[0])
    group_2 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    group_3 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])

    concatenated_1 = Concatenate(axis=-1)([group_1, group_2, group_3])

    # Second Block with multiple branches
    branch_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(concatenated_1)

    branch_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(concatenated_1)
    branch_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch_2)

    branch_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(concatenated_1)
    branch_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch_3)

    branch_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concatenated_1)
    branch_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch_4)

    concatenated_2 = Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])

    # Flatten and Fully Connected Layer
    flatten = Flatten()(concatenated_2)
    outputs = Dense(10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()