import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    split1 = Lambda(lambda x: keras.backend.split(x,3,axis=-1))(input_layer)
    split2 = Lambda(lambda x: keras.backend.split(x,3,axis=-1))(input_layer)
    split3 = Lambda(lambda x: keras.backend.split(x,3,axis=-1))(input_layer)

    # Process each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split2[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split3[2])

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    dropout1 = Dropout(0.5)(pool1)
    dropout2 = Dropout(0.5)(pool2)
    dropout3 = Dropout(0.5)(pool3)

    concat = Concatenate()([dropout1, dropout2, dropout3])

    conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(concat)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    dropout4 = Dropout(0.5)(pool4)
    dropout5 = Dropout(0.5)(pool5)

    # Branch pathway
    conv6 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(concat)
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(concat)

    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

    dropout6 = Dropout(0.5)(pool6)
    dropout7 = Dropout(0.5)(pool7)

    # Concatenate the outputs from the main pathway and the branch pathway
    concat_branch = Concatenate()([conv4, conv6])

    # Fully connected layer
    dense = Dense(units=128, activation='relu')(concat_branch)
    output = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=[input_layer, split1[1], split2[0], split3[1]], outputs=output)

    return model