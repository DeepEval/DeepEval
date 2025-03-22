import keras
from keras.layers import Lambda, Conv2D, Concatenate, Add, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    input_shape = (32, 32, 3)

    # Main path
    x1 = Lambda(lambda x: tf.split(x, 3, axis=1))(input_shape)
    x1 = Conv2D(32, (1, 1), activation='relu')(x1)
    x2 = Conv2D(64, (3, 3), activation='relu')(x1)
    x3 = Conv2D(128, (5, 5), activation='relu')(x1)
    x = Concatenate()([x1, x2, x3])

    # Branch path
    x = Conv2D(128, (1, 1), activation='relu')(x)

    # Fusion
    x = Add()([x, x])

    # Classifier
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.models.Model(inputs=input_shape, outputs=x)

    return model