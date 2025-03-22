from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, concatenate, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the VGG16 layers
    for layer in vgg.layers:
        layer.trainable = False

    # Add the first part of the model: convolutional layers with max pooling
    x = vgg.output
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Add the second part of the model: convolutional layers with dropout
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)

    # Add the third part of the model: upsampling layers with skip connections
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, vgg.layers[5].output])
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, vgg.layers[10].output])
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, vgg.layers[15].output])
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)

    # Add the final layer for the 10-class classification
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    # Create the model
    model = Model(inputs=vgg.input, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model