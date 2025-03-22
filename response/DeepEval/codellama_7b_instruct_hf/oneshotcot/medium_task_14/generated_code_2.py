import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new input layer with a 32x32 image size
    input_layer = Input(shape=(32, 32, 3))

    # Add the new input layer to the base model
    x = base_model(input_layer)

    # Add three sequential blocks, each comprising a convolutional layer, a batch normalization layer, and a ReLU activation function
    x1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x3)
    x3 = BatchNormalization()(x3)

    # Add the three sequential blocks to the base model
    x = Concatenate()([x1, x2, x3])

    # Add a parallel branch of convolutional layers to process the input directly
    x_parallel = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x_parallel = BatchNormalization()(x_parallel)
    x_parallel = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x_parallel)
    x_parallel = BatchNormalization()(x_parallel)
    x_parallel = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x_parallel)
    x_parallel = BatchNormalization()(x_parallel)

    # Add the parallel branch to the base model
    x = Concatenate()([x, x_parallel])

    # Add a flatten layer and two fully connected layers for classification
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    return model