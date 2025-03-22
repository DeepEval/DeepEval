import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16

# Load the pre-trained VGG16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the VGG16 layers
for layer in vgg.layers:
    layer.trainable = False


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the multi-branch convolutional layers
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (1, 7), activation='relu')(input_layer)
    branch3 = Conv2D(32, (7, 1), activation='relu')(input_layer)
    branch4 = Conv2D(32, (7, 7), activation='relu')(input_layer)

    # Define the fusion layer
    fusion = Concatenate()([branch1, branch2, branch3, branch4])

    # Define the pooling layers
    pool1 = MaxPooling2D((2, 2))(fusion)
    pool2 = AveragePooling2D((2, 2))(fusion)

    # Define the flatten layer
    flatten = Flatten()(pool1)

    # Define the fully connected layers
    fc1 = Dense(64, activation='relu')(flatten)
    fc2 = Dense(32, activation='relu')(fc1)
    output = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model