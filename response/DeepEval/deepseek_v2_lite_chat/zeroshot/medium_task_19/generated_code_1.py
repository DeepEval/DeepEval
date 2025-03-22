import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input shape should match the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    img_input = Input(shape=input_shape)

    # First branch: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(64, (1, 1), activation='relu')(img_input)

    # Second branch: 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(64, (1, 1), activation='relu')(branch1)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    # Third branch: 1x1 convolution followed by a 5x5 convolution
    branch3 = Conv2D(128, (1, 1), activation='relu')(branch1)
    branch3 = Conv2D(128, (5, 5), activation='relu')(branch3)

    # Fourth branch: 3x3 max pooling followed by a 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3))(img_input)
    branch4 = Conv2D(256, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs of the branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten the concatenated tensor
    flat = Flatten()(concat)

    # Fully connected layers for classification
    output1 = Dense(512, activation='relu')(flat)
    output2 = Dense(256, activation='relu')(output1)
    output3 = Dense(128, activation='relu')(output2)
    output4 = Dense(10, activation='softmax')(output3)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=img_input, outputs=output4)

    # Optional: load pre-trained model weights to initialize the layers
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model.layers.pop()  # Remove previous fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# If you want to use this model for inference, make sure to compile and train it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=epochs)