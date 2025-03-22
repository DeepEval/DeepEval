import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: 1x1 convolution
    conv1_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1_path = BatchNormalization()(conv1_path)

    # Path 2: sequence of convolutions
    conv2_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm2_path = BatchNormalization()(conv2_path)
    conv3_path = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(batch_norm2_path)
    batch_norm3_path = BatchNormalization()(conv3_path)
    conv4_path = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm3_path)
    batch_norm4_path = BatchNormalization()(conv4_path)

    # Concatenate the outputs of the two paths and apply a 1x1 convolution
    concat_path = Concatenate()([conv1_path, batch_norm4_path])
    conv5_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat_path)
    batch_norm5_path = BatchNormalization()(conv5_path)

    # Create a branch that connects directly to the input
    branch = Add()([input_layer, batch_norm5_path])

    # Merge the outputs of the main path and the branch through addition
    merge = Add()([branch, batch_norm5_path])

    # Flatten the merged output
    flatten_layer = Flatten()(merge)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = dl_model()
model.summary()