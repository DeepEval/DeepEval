from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Reshape
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define input shape
    input_shape = (28, 28, 1)

    # Define the first block
    block1 = Input(shape=input_shape)
    x = block1
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the second block
    block2 = Input(shape=input_shape)
    x = block2
    x = Lambda(lambda x: tf.split(x, 4, axis=-1))(x)
    x = Concatenate()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[block1, block2], outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model