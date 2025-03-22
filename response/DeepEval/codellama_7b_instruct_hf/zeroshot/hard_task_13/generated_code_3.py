from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block of the model
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_1)
    conv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_2)
    pool1 = MaxPooling2D((2, 2))(conv1_3)
    flat1 = Flatten()(pool1)

    # Define the second block of the model
    conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(flat1)
    conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_1)
    conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_2)
    pool2 = MaxPooling2D((2, 2))(conv2_3)
    flat2 = Flatten()(pool2)
    dense1 = Dense(1024, activation='relu')(flat2)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=input_shape, outputs=dense2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model