import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input



def dl_model():
    # load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # preprocess the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # one-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # define the input shape
    input_shape = (32, 32, 3)

    # define the model architecture
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # define the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    return model