import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.models import Model
from keras.applications import VGG16

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first block
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = Conv2D(32, (1, 1), activation='relu')(x[0])
    x = Conv2D(32, (3, 3), activation='relu')(x[1])
    x = Conv2D(32, (5, 5), activation='relu')(x[2])
    x = Dropout(0.25)(x)
    x = Concatenate()([x[0], x[1], x[2]])

    # Define the second block
    x = Lambda(lambda x: tf.split(x, 4, axis=3))(x)
    x = Conv2D(64, (1, 1), activation='relu')(x[0])
    x = Conv2D(64, (3, 3), activation='relu')(x[1])
    x = Conv2D(64, (5, 5), activation='relu')(x[2])
    x = MaxPooling2D((2, 2))(x[3])
    x = Conv2D(64, (1, 1), activation='relu')(x[0])
    x = Conv2D(64, (3, 3), activation='relu')(x[1])
    x = Conv2D(64, (5, 5), activation='relu')(x[2])
    x = Dropout(0.5)(x)
    x = Concatenate()([x[0], x[1], x[2]])

    # Define the output layer
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model