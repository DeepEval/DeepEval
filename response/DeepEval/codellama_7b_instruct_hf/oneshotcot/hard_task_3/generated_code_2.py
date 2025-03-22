import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16


def dl_model():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255


    input_shape = (32, 32, 3)


    input_layer = Input(shape=input_shape)
    groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)


    conv1 = Conv2D(32, (1, 1), activation='relu')(groups[0])
    conv2 = Conv2D(32, (3, 3), activation='relu')(groups[1])
    conv3 = Conv2D(32, (3, 3), activation='relu')(groups[2])


    dropout = Dropout(0.5)(conv3)


    main_pathway = Concatenate()([conv1, conv2, conv3])


    branch_pathway = Conv2D(32, (1, 1), activation='relu')(input_layer)


    output = Add()([main_pathway, branch_pathway])


    flatten = Flatten()(output)
    fc = Dense(10, activation='softmax')(flatten)


    model = Model(inputs=input_layer, outputs=fc)


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


    model.evaluate(x_test, y_test)

    return model