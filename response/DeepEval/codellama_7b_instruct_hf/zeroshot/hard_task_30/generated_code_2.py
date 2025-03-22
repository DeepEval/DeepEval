import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Dual-path structure with main and branch paths
    main_path = Conv2D(64, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    branch_path = input_layer

    # Addition of main and branch paths
    x = Add()([main_path, branch_path])

    # Block 2
    # Splits the input into three groups along the channel
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)

    # Feature extraction using depthwise separable convolutional layers with different kernel sizes
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x[0])
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x[1])
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x[2])

    # Concatenation of outputs from three groups
    x = Concatenate()(x)

    # Flatten and dropout
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_layer, outputs=outputs)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model