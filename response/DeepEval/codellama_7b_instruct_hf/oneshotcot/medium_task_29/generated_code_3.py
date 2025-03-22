import keras
from keras.layers import Input, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape and number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # Load the VGG-16 pre-trained model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Add a new input layer and max pooling layers with varying window sizes
    input_layer = Input(shape=input_shape)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(input_layer)

    # Flatten the output of each pooling layer and concatenate them
    pool1_flatten = Flatten()(pool1)
    pool2_flatten = Flatten()(pool2)
    pool3_flatten = Flatten()(pool3)
    concat_pool = Concatenate()([pool1_flatten, pool2_flatten, pool3_flatten])

    # Add batch normalization and fully connected layers
    batch_norm = BatchNormalization()(concat_pool)
    dense1 = Dense(units=128, activation='relu')(batch_norm)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=num_classes, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model