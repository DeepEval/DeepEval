import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle

 和 return model
def dl_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


    input_shape = (32, 32, 3)
    num_classes = 10


    input_layer = Input(shape=input_shape)

    # Main pathway
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    dropout1 = Dropout(0.2)(maxpool1)

    # Branch pathway
    conv4 = Conv2D(128, (1, 1), activation='relu')(input_layer)

    # Combine the main and branch pathways
    add = Add()([maxpool1, conv4])
    flatten = Flatten()(add)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))


    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)


    predictions = model.predict(x_test)

    return model