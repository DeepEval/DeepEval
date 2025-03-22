import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout, Add, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def extract_features(input_tensor):
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_tensor)
        features = []
        for split_part in split_layer:
            conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_part)
            conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
            dropout1 = Dropout(rate=0.2)(pool1)
            features.append(dropout1)
        return features

    main_features = extract_features(input_tensor=input_layer)
    branch_features = extract_features(input_tensor=input_layer)

    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_features[0])
    branch_path = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_features[0])
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(main_path)
    branch_path = Flatten()(branch_path)
    main_path = Flatten()(main_path)

    main_output = Dense(units=256, activation='relu')(main_path)
    main_output = Dropout(rate=0.5)(main_output)
    branch_output = Dense(units=256, activation='relu')(branch_path)
    branch_output = Dropout(rate=0.5)(branch_output)

    output = Add()([main_output, branch_output])
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])