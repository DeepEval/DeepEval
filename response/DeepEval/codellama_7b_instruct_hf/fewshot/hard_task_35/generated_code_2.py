import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, Reshape
from keras.models import Model

 和 return model
def dl_model():

    input_shape = (32, 32, 3)


    branch_1 = GlobalAveragePooling2D()(input_shape)
    branch_1 = Dense(units=64, activation='relu')(branch_1)
    branch_1 = Dense(units=128, activation='relu')(branch_1)
    branch_1 = Flatten()(branch_1)


    branch_2 = GlobalAveragePooling2D()(input_shape)
    branch_2 = Dense(units=64, activation='relu')(branch_2)
    branch_2 = Dense(units=128, activation='relu')(branch_2)
    branch_2 = Flatten()(branch_2)


    model = Model(inputs=input_shape, outputs=branch_1, branch_2)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=10, batch_size=32)


    model.evaluate(X_test, y_test)


    predictions = model.predict(X_test)

    return model