from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense, Reshape
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define input shape and number of classes
    input_shape = (28, 28, 1)
    num_classes = 10

    # Define the first block
    first_block = Model(inputs=Input(shape=input_shape),
                        outputs=Flatten()(AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_shape)))

    # Define the second block
    second_block = Model(inputs=Input(shape=input_shape),
                         outputs=Concatenate(axis=1)([Dense(units=128, activation='relu')(Dropout(0.2)(first_block)) for _ in range(4)]))

    # Define the model
    model = Model(inputs=first_block, outputs=second_block)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model