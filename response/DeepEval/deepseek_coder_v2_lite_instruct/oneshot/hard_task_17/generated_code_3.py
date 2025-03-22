import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=32, activation='relu')(avg_pool)
        fc2 = Dense(units=32, activation='relu')(fc1)
        reshape = keras.layers.Reshape((32, 32, 1))(fc2)
        weighted_features = keras.layers.multiply([input_tensor, reshape])
        return weighted_features

    block1_output = block1(input_layer)

    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    # Branch from Block 1
    block1_branch = keras.layers.GlobalAveragePooling2D()(block1_output)

    # Fuse the outputs
    fused_output = keras.layers.Add()([max_pool, block1_branch])

    # Classification
    fc1 = Dense(units=64, activation='relu')(fused_output)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=fc2)

    return model

# Example usage:
# model = dl_model()
# model.summary()