import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet
from keras.applications.inception_v3 import InceptionV3


def dl_model():

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


    input_shape = (32, 32, 3)


    num_classes = 10


    num_branches = 3


    num_filters_branch_1 = 16
    num_filters_branch_2 = 32
    num_filters_branch_3 = 64


    num_layers_branch_1 = 2
    num_layers_branch_2 = 3
    num_layers_branch_3 = 4


    dropout_rate = 0.2


    activation_branch_1 = 'relu'
    activation_branch_2 = 'relu'
    activation_branch_3 = 'relu'


    pool_size = (2, 2)


    batch_norm_params = {'axis': -1, 'momentum': 0.9, 'epsilon': 1e-05, 'center': True, 'scale': True, 'beta_initializer': 'zeros', 'gamma_initializer': 'ones', 'moving_mean_initializer': 'zeros', 'moving_variance_initializer': 'ones'}


    dense_params = {'units': 128, 'activation': 'relu', 'kernel_initializer': 'he_normal', 'bias_initializer': 'zeros', 'kernel_regularizer': 'l2', 'bias_regularizer': 'l2', 'activity_regularizer': 'l2', 'kernel_constraint': 'max_norm', 'bias_constraint': 'max_norm'}


    inputs = Input(shape=input_shape, name='input')

    branch_1 = Conv2D(filters=num_filters_branch_1, kernel_size=(3, 3), activation=activation_branch_1, padding='same')(inputs)
    branch_2 = Conv2D(filters=num_filters_branch_2, kernel_size=(3, 3), activation=activation_branch_2, padding='same')(inputs)
    branch_3 = Conv2D(filters=num_filters_branch_3, kernel_size=(3, 3), activation=activation_branch_3, padding='same')(inputs)

    dropout_1 = Dropout(dropout_rate)(branch_1)
    dropout_2 = Dropout(dropout_rate)(branch_2)
    dropout_3 = Dropout(dropout_rate)(branch_3)

    branch_1_pool = MaxPooling2D(pool_size=pool_size)(dropout_1)
    branch_2_pool = MaxPooling2D(pool_size=pool_size)(dropout_2)
    branch_3_pool = MaxPooling2D(pool_size=pool_size)(dropout_3)

    branch_1_flatten = Flatten()(branch_1_pool)
    branch_2_flatten = Flatten()(branch_2_pool)
    branch_3_flatten = Flatten()(branch_3_pool)

    branch_1_dense = Dense(**dense_params)(branch_1_flatten)
    branch_2_dense = Dense(**dense_params)(branch_2_flatten)
    branch_3_dense = Dense(**dense_params)(branch_3_flatten)

    output = Concatenate(axis=1)([branch_1_dense, branch_2_dense, branch_3_dense])
    output = BatchNormalization(**batch_norm_params)(output)
    output = Flatten()(output)
    output = Dense(**dense_params)(output)
    output = Dense(**dense_params)(output)
    output = Dense(**dense_params)(output)


    model = Model(inputs=inputs, outputs=output, name='cifar10_branches')


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))


    model.evaluate(X_test, y_test)

    return model