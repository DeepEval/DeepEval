import keras
from keras.layers import Input, Lambda, Concatenate, Dense, Flatten
from keras.applications import VGG16

def dl_model():
    # Load the VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the VGG16 layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Create the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Extract features from each channel group using separable convolutional layers
    channel_groups = [
        Dense(64, activation='relu', kernel_initializer='he_normal')(channel_groups),
        Dense(128, activation='relu', kernel_initializer='he_normal')(channel_groups),
        Dense(256, activation='relu', kernel_initializer='he_normal')(channel_groups),
        Dense(512, activation='relu', kernel_initializer='he_normal')(channel_groups)
    ]

    # Concatenate the outputs from each channel group
    channel_groups = Concatenate()(channel_groups)

    # Flatten the output and add dense layers
    flatten_layer = Flatten()(channel_groups)
    dense_layer1 = Dense(128, activation='relu')(flatten_layer)
    dense_layer2 = Dense(64, activation='relu')(dense_layer1)
    dense_layer3 = Dense(32, activation='relu')(dense_layer2)
    output_layer = Dense(10, activation='softmax')(dense_layer3)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model