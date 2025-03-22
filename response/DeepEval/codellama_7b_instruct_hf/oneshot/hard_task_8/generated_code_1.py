import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Flatten, Dense, Permute
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

，增加dl_model 和 return model
def dl_model():

    input_shape = (28, 28, 1)
    num_classes = 10


    def block_1(input_tensor):
        # 1x1 convolution
        conv1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        # depthwise separable convolution
        depthwise_conv = DepthwiseSeparableConv2D(32, (3, 3), activation='relu')(conv1)
        # 1x1 convolution
        conv2 = Conv2D(32, (1, 1), activation='relu')(depthwise_conv)
        # concatenate features from both paths
        concatenated = Concatenate()([conv1, conv2])
        return concatenated


    def block_2(input_tensor):
        # reshape input tensor to (height, width, groups, channels_per_group)
        reshaped = Permute((2, 3, 1, 4))(input_tensor)
        # swap the third and fourth dimensions
        swapped = Permute((2, 3, 4, 1))(reshaped)
        # reshape back to original shape
        reshaped_back = Permute((2, 3, 1, 4))(swapped)
        return reshaped_back


    input_layer = Input(shape=input_shape)
    block_1_output = block_1(input_layer)
    block_2_output = block_2(block_1_output)
    flattened = Flatten()(block_2_output)
    output_layer = Dense(num_classes, activation='softmax')(flattened)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# generate image data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'path/to/train/directory',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'path/to/test/directory',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator))