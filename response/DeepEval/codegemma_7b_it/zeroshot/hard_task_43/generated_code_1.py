from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():

    input_shape = (28, 28, 1)

    # Block 1
    input_img = layers.Input(shape=input_shape)
    branch_1x1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(input_img)
    branch_2x2 = layers.Conv2D(filters=32, kernel_size=2, strides=2, activation='relu')(input_img)
    branch_4x4 = layers.Conv2D(filters=32, kernel_size=4, strides=4, activation='relu')(input_img)
    concat_1 = layers.Concatenate()([branch_1x1, branch_2x2, branch_4x4])
    flatten_1 = layers.Flatten()(concat_1)

    # Block 2
    concat_2 = layers.Reshape((7, 7, 128))(flatten_1)
    branch_conv_1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(concat_2)
    branch_conv_2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(concat_2)

    branch_conv_3 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(concat_2)
    branch_conv_3 = layers.Conv2D(filters=32, kernel_size=7, activation='relu')(branch_conv_3)
    branch_conv_3 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(branch_conv_3)

    branch_conv_4 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(concat_2)
    branch_avg_pool = layers.AveragePooling2D(pool_size=(2, 2))(concat_2)
    concat_3 = layers.Concatenate()([branch_conv_1, branch_conv_2, branch_conv_3, branch_conv_4, branch_avg_pool])

    # Classification
    flatten_2 = layers.Flatten()(concat_3)
    output = layers.Dense(units=10, activation='softmax')(flatten_2)

    model = Model(inputs=input_img, outputs=output)

    return model

model = dl_model()