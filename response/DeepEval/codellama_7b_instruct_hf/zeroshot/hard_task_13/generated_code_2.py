from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16


def dl_model():
    # Load the VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the layers of the VGG16 model
    for layer in vgg_model.layers:
        layer.trainable = False

    # Add the custom layers to the model
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(16, (3, 3), activation='relu')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Print the summary of the model
    print(model.summary())

    return model