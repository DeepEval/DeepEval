from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, DepthwiseSeparableConv2D, concatenate, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the pre-trained VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Get the input tensor and add a lambda layer to split it into three groups
    input_tensor = Input(shape=(32, 32, 3))
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)

    # Extract features from each group using depthwise separable convolutional layers
    x1 = Conv2D(64, (1, 1), activation='relu')(x[0])
    x1 = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')(x1)
    x1 = DepthwiseSeparableConv2D(64, (5, 5), activation='relu')(x1)

    x2 = Conv2D(64, (1, 1), activation='relu')(x[1])
    x2 = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')(x2)
    x2 = DepthwiseSeparableConv2D(64, (5, 5), activation='relu')(x2)

    x3 = Conv2D(64, (1, 1), activation='relu')(x[2])
    x3 = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')(x3)
    x3 = DepthwiseSeparableConv2D(64, (5, 5), activation='relu')(x3)

    # Concatenate the outputs of the three groups
    x = concatenate([x1, x2, x3])

    # Flatten the concatenated output and pass it through a fully connected layer
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=x)

    # Freeze the weights of the VGG16 model
    for layer in vgg16.layers:
        layer.trainable = False

    # Add the custom layers to the model
    model.add(x1)
    model.add(x2)
    model.add(x3)

    # Return the constructed model
    return model