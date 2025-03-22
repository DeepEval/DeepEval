import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

    # Define the second block
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    vgg16.trainable = False
    output_tensor = vgg16(output_tensor)
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(128, activation='relu')(output_tensor)
    output_tensor = Dense(10, activation='softmax')(output_tensor)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model