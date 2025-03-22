from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Apply 1x1 convolutions to each group independently
    groups = [Conv2D(filters=int(x.shape[3] / 3), kernel_size=1, activation='relu')(x) for x in groups]

    # Downsample each group using average pooling
    groups = [AveragePooling2D(pool_size=2)(x) for x in groups]

    # Concatenate the three groups along the channel dimension
    groups = tf.concat(groups, axis=3)

    # Flatten the concatenated feature maps
    flattened = Flatten()(groups)

    # Pass the flattened feature maps through two fully connected layers for classification
    classifier = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=input_layer, outputs=classifier)

    return model