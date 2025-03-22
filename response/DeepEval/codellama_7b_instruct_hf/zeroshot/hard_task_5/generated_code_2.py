from keras.layers import Input, Conv2D, Lambda, Concatenate, Permute, Reshape, Add, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    input_shape = (32, 32, 3)
    # Main path
    # Split input into three groups using Lambda layer
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(Input(shape=input_shape))
    # Apply 1x1 convolutional layers to each group
    x1 = Conv2D(64, (1, 1), activation='relu')(x[0])
    x2 = Conv2D(64, (1, 1), activation='relu')(x[1])
    x3 = Conv2D(64, (1, 1), activation='relu')(x[2])
    # Concatenate feature maps from each group
    x = Concatenate()([x1, x2, x3])
    # Apply 3x3 depthwise separable convolution
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    # Apply channel shuffle
    x = Permute((1, 2, 3, 0))(x)
    x = Reshape((-1, 128))(x)
    # Add branch
    x_branch = Conv2D(128, (1, 1), activation='relu')(Input(shape=input_shape))
    x = Add()([x, x_branch])
    # Apply final layers
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=Input(shape=input_shape), outputs=x)
    return model