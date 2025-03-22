import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split the input into three groups along the channel dimension
    input_shape = x_train.shape[1:]
    input_layer = Input(shape=input_shape)

    # Split the input into three groups
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group
    split1 = [Conv2D(filters=input_shape[i] // 3, kernel_size=(1, 1), activation='relu')(x) for i, x in enumerate(split1)]

    # Downsample each group using average pooling
    split1 = [MaxPooling2D(pool_size=(2, 2))(x) for x in split1]

    # Concatenate the feature maps along the channel dimension
    concat = Concatenate(axis=-1)(split1)

    # Flatten and pass through two fully connected layers for classification
    flat = Flatten()(concat)
    output_layer = Dense(10, activation='softmax')(flat)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and return the model
model = dl_model()