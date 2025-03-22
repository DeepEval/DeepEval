from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense
from keras.applications.vgg16 import VGG16


def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Load the VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the VGG16 layers
    for layer in vgg.layers:
        layer.trainable = False

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first block of the model
    conv_layer1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv_layer1)
    conv_layer2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batch_norm1)
    batch_norm2 = BatchNormalization()(conv_layer2)
    conv_layer3 = Conv2D(32, (3, 3), activation='relu', padding='same')(batch_norm2)
    batch_norm3 = BatchNormalization()(conv_layer3)

    # Define the second block of the model
    conv_layer4 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm3)
    batch_norm4 = BatchNormalization()(conv_layer4)
    conv_layer5 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm4)
    batch_norm5 = BatchNormalization()(conv_layer5)
    conv_layer6 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm5)
    batch_norm6 = BatchNormalization()(conv_layer6)

    # Define the third block of the model
    conv_layer7 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch_norm6)
    batch_norm7 = BatchNormalization()(conv_layer7)
    conv_layer8 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch_norm7)
    batch_norm8 = BatchNormalization()(conv_layer8)
    conv_layer9 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch_norm8)
    batch_norm9 = BatchNormalization()(conv_layer9)

    # Define the fully connected layers
    flatten = Flatten()(conv_layer9)
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model