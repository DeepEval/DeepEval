import keras
from keras.layers import Input, Reshape, Permute, Dense, Softmax
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    reshaped_input = Reshape(target_shape=(32, 32, 3, 1))(input_layer)
    permuted_input = Permute((2, 3, 1))(reshaped_input)
    reshaped_input = Reshape(target_shape=(32, 32, 3))(permuted_input)
    conv1 = Conv2D(32, (3, 3), activation='relu')(reshaped_input)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    flattened = Flatten()(pool3)
    output = Dense(10, activation='softmax')(flattened)
    model = Model(inputs=input_layer, outputs=output)
    return model