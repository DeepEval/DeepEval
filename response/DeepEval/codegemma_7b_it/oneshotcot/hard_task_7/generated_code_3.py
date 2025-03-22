import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    # Block 1
    def block1(input_tensor):
        group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal', depthwise_constraint='max_norm')(group1)
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        group2 = Lambda(lambda x: tf.identity(x))(group1)
        output_tensor = Concatenate()([group1, group2])
        return output_tensor
    block1_output = block1(input_tensor=conv)

    # Block 2
    def block2(input_tensor):
        batch_shape = tf.shape(input_tensor)
        input_tensor = Reshape(target_shape=(batch_shape[1], batch_shape[2], 1, batch_shape[3]))(input_tensor)
        input_tensor = Permute(dims=(0, 1, 3, 2))(input_tensor)
        input_tensor = Reshape(target_shape=(batch_shape[1], batch_shape[2], batch_shape[3]))(input_tensor)
        return input_tensor
    block2_output = block2(input_tensor=block1_output)

    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model

# Example usage:
model = dl_model()