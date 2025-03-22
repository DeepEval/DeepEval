import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


input_layer = Input(shape=(28, 28, 1))


def block(input_tensor):
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    output_tensor = Concatenate()([conv1, conv2, conv3, max_pooling])
    return output_tensor


def branch_block(input_tensor):
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    output_tensor = Concatenate()([conv1, conv2, conv3, max_pooling])
    return output_tensor


main_path = Input(shape=(28, 28, 1))
main_block_output = block(main_path)


branch_path = Input(shape=(28, 28, 1))
branch_block_output = branch_block(branch_path)


def final_block(input_tensor):
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    output_tensor = Concatenate()([conv1, conv2, conv3, max_pooling])
    return output_tensor


model = keras.Model(inputs=[main_path, branch_path], outputs=final_block(main_block_output + branch_block_output))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))