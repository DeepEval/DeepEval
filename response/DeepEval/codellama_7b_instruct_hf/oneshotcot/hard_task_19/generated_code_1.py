import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model


input_shape = (32, 32, 3)


main_path = Sequential()
main_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
main_path.add(Conv2D(64, (3, 3), activation='relu'))
main_path.add(MaxPooling2D((2, 2)))
main_path.add(Conv2D(128, (3, 3), activation='relu'))
main_path.add(MaxPooling2D((2, 2)))
main_path.add(Flatten())
main_path.add(Dense(128, activation='relu'))


branch_path = Sequential()
branch_path.add(GlobalAveragePooling2D())
branch_path.add(Dense(32, activation='relu'))
branch_path.add(Dense(16, activation='relu'))


concat_layer = Concatenate()([main_path, branch_path])


batch_norm = BatchNormalization()(concat_layer)
flatten_layer = Flatten()(batch_norm)


output_layer = Dense(10, activation='softmax')(flatten_layer)


model = Model(inputs=main_path.input, outputs=output_layer)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))