from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    main_split = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_input)
    main_conv1 = Conv2D(64, (3, 3), activation='relu')(main_split[0])
    main_conv2 = Conv2D(64, (3, 3), activation='relu')(main_split[1])
    main_conv3 = Conv2D(64, (3, 3), activation='relu')(main_split[2])
    main_output = Concatenate()([main_conv1, main_conv2, main_conv3])
    main_output = MaxPooling2D(pool_size=(2, 2))(main_output)
    main_output = Flatten()(main_output)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_conv1 = Conv2D(64, (1, 1), activation='relu')(branch_input)
    branch_output = MaxPooling2D(pool_size=(2, 2))(branch_conv1)
    branch_output = Flatten()(branch_output)

    # Fusion
    fusion_output = Add()([main_output, branch_output])
    fusion_output = Dense(10, activation='softmax')(fusion_output)

    model = Model(inputs=[main_input, branch_input], outputs=fusion_output)

    return model