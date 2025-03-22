import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
from keras.applications.mobilenet_v2 import SeparableConv2D

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))  

    # Split the input along the last dimension
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)

    # Feature extraction for each group
    group1 = SeparableConv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    group3 = SeparableConv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])

    # Concatenate the outputs of the groups
    concat_tensor = Concatenate()( [group1, group2, group3])

    # Flatten and classify
    flatten_layer = Flatten()(concat_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)

    return model