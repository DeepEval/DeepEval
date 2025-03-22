import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Permute, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)  # Assuming input image shape is (32, 32, 3) for CIFAR-10
    input_layer = Input(shape=input_shape)

    # Reshape input into 3 groups
    reshaped = Reshape((input_shape[0] * input_shape[1] * input_shape[2],))(input_layer)
    split_tensors = keras.layers.Lambda(lambda x: keras.layers.Reshape((x.shape[0] // 3, 3, x.shape[-1]))(x))(reshaped)

    # Swap third and fourth dimensions for channel shuffling
    permuted = Permute((3, 1, 2))(split_tensors)

    # Reshape back to original input shape
    reshaped_permuted = Reshape(input_shape)(permuted)

    # Add fully connected layer
    dense = Dense(units=128, activation='relu')(reshaped_permuted)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model