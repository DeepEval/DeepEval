import keras
from keras.layers import Input, Reshape, Permute, Lambda, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    input_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor=input_layer)
    reshaped = Reshape(target_shape=(32, 32, 1, 3))(input_tensor=input_layer)
    permuted = Permute(dims=[3, 1, 2, 4])(reshaped)
    reshaped = Reshape(target_shape=(32, 32, 3))(permuted)
    output_layer = Dense(units=10, activation='softmax')(reshaped)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model