import keras
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Load the VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Get the last convolutional layer of the VGG16 model
    conv_layer = vgg.layers[-1]

    # Create a new model that consists of the global average pooling layer
    # followed by two fully connected layers
    model = keras.models.Sequential([
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Freeze the weights of the VGG16 model
    # so that they are not updated during training
    for layer in vgg.layers:
        layer.trainable = False

    # Add the convolutional layer to the new model
    model.add(conv_layer)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model