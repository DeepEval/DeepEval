import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Flatten
from keras.optimizers import Adam


input_shape = (32, 32, 3)


def first_block(input_tensor, filters1, filters2, pool_size=(2, 2)):
    x = Conv2D(filters=filters1, kernel_size=(3, 3), activation='relu')(input_tensor)
    x = Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu')(x)
    main_output = x
    identity_output = input_tensor
    return Add()([main_output, identity_output])


def second_block(input_tensor, n_classes):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    return x


def dl_model():
    # Input tensor for CIFAR-10 dataset
    inputs = Input(shape=input_shape)
    
    # First block
    x = first_block(inputs, 32, 32)
    x = first_block(x, 64, 64)
    
    # Second block
    x = second_block(x, 10)
    
    # Create the model
    model = Model(inputs=inputs, outputs=x)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model