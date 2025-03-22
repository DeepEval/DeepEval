from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    input_shape = (28, 28, 1)
    num_classes = 10

    # First block
    x = Input(shape=input_shape)
    x = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Second block
    x = Lambda(lambda x: tf.split(x, 4, -1))(x)
    for i in range(4):
        x = Conv2D(64, (1, 1), activation='relu', padding='same')(x[i])
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x[i])
        x = Conv2D(64, (5, 5), activation='relu', padding='same')(x[i])
        x = Conv2D(64, (7, 7), activation='relu', padding='same')(x[i])
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x[i])
        x = Lambda(lambda x: tf.concat(x, -1))(x[i])

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=x, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model