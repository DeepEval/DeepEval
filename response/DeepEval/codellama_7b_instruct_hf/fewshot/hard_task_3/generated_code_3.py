import keras
from keras.layers import Input, Lambda, Concatenate, Flatten, Dense, Dropout
from keras.applications import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    main_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    branch_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))(main_path)
    conv2 = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))(branch_path)
    concatenate = Concatenate()([conv1, conv2])
    flatten = Flatten()(concatenate)
    dropout = Dropout(0.2)(flatten)
    dense = Dense(units=10, activation='softmax')(dropout)
    model = keras.Model(inputs=input_layer, outputs=dense)
    return model