import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.applications import VGG16

def dl_model():
    # Load the pre-trained VGG16 model except the last layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Assuming the CIFAR-10 data is already preprocessed and normalized
    input_tensor = Input(shape=(32, 32, 3))
    x = base_model(input_tensor, training=False)

    # Split the model into three parts based on channel (split at the second dimension)
    split1 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    split2 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    split3 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)

    # Convolutional layers for feature extraction
    x1 = split1[0]
    x2 = split1[1]
    x3 = split1[2]
    x = concatenate([x1, x2, x3])
    x = Conv2D(128, 3, activation='relu')(x)  # 3x3 convolution
    x = MaxPooling2D()(x)
    x = split2[0]
    x = concatenate([x, split2[1], split2[2]])
    x = Conv2D(64, 3, activation='relu')(x)  # series of 1x1, 3x3 convolutions
    x = MaxPooling2D()(x)
    x = split3[0]
    x = concatenate([x, split3[1], split3[2]])
    x = Conv2D(32, 3, activation='relu')(x)  # max pooling branch
    x = MaxPooling2D()(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the final model
    model = Model(input_tensor, output)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    return model

# Build the model
model = dl_model()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])