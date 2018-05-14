import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from mdn_functions import *


if K.backend() == 'tensorflow':
    print('Backend is TensorFlow')
    K.set_image_dim_ordering('tf')
elif K.backend() == 'theano':
    print('Backend is Theano.')
    K.set_image_dim_ordering('th')
else:
    raise Exception('Unhandled Backend found: ', K.backend())


def create_model(component, weights_path=None):
    # custom layers
    inputs = Input(shape=(2, ))
    net = Dense(5, activation="relu")(inputs)
    net = Dense(10, activation="tanh")(net)
    net = Dense(5, activation="relu")(net)
    net = add_univariate_mixture_layer(net, component)
    net = Model(inputs=inputs, outputs=net)

    # set up optimizer
    optimizer = Adam(lr=0.01)

    net.compile(loss=negative_log_likelihood_loss(component), optimizer=optimizer)
    print (net.summary())

    if weights_path:
        net.load_weights(weights_path)

    return net


def pearson_correlation(x, y):
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt((K.sum(K.square(xm)) * K.sum(K.square(ym))))
    r = r_num / (r_den + 1e-10)
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


if __name__ == "__main__":
    epochs = 30
    batches = 128
    training_data_path = '../data/dataset-00/noisy_data_00.csv'
    testing_data_path = '../data/dataset-00/test_noisy_data_00.csv'
    # training_data_path = '../data/rpal_force_derived/assessment/m25_input.npy'
    # validation_data_path = '../data/rpal_force_derived/assessment/m25_output.npy'
    # testing_data_path = '../data/rpal_force_derived/assessment/test_m25_input.npy'
    # testing_val_data_path = '../data/rpal_force_derived/assessment/test_m25_output.npy'

    # read and process data
    dataset = pd.read_csv(training_data_path).values
    test_dataset = pd.read_csv(testing_data_path).values
    # training_data = np.load(training_data_path)
    # valid_data = np.load(validation_data_path)
    # testing_data = np.load(testing_data_path)
    # testing_valid_data = np.load(testing_val_data_path)

    # training_data, testing_data, training_range_data, testing_range_data = train_test_split(dataset[:, 2:], dataset[:, 0], test_size=0.25)
    training_data, testing_data, training_bearing_data, testing_bearing_data = train_test_split(dataset[:, 2:], dataset[:, 1], test_size=0.25)
    # training_data, testing_input_data, training_output_data, testing_output_data = train_test_split(training_data, valid_data, test_size=0.25)

    testing_data = test_dataset[:, 2:]
    testing_range_data = test_dataset[:, 0]
    testing_bearing_data = test_dataset[:, 1]

    # create model
    model = create_model(1)

    kf = KFold(n_splits=10)
    kf.get_n_splits(training_data)

    for train_index, validation_index in kf.split(training_data):
        train, validation = training_data[train_index], training_data[validation_index]
        # range_train, range_validation = training_range_data[train_index], training_range_data[validation_index]
        bearing_train, bearing_validation = training_bearing_data[train_index], training_bearing_data[validation_index]
        # motion_train, motion_validation = training_output_data[train_index], training_output_data[validation_index]

        # history = model.fit(train, range_train, validation_data=(validation, range_validation), epochs=epochs, batch_size=batches)
        history = model.fit(train, bearing_train, validation_data=(validation, bearing_validation), epochs=epochs, batch_size=batches)
        # history = model.fit(train, motion_train, validation_data=(validation, motion_validation), epochs=epochs, batch_size=batches)

        # score = model.evaluate(testing_data, testing_range_data, batch_size=batches)
        score = model.evaluate(testing_data, testing_bearing_data, batch_size=batches)
        # score = model.evaluate(testing_data, testing_valid_data, batch_size=batches)

        print(score)

        test_for = testing_bearing_data

        result = model.predict(testing_data)
        alpha, mu, sigma = separate_mixture_matrix_into_parameters(mdn_output_matrix=result, nb_components=1)
        means, stdvs = compute_mixture_total_mean_variance(mix_coeff_matrix=alpha, means_matrix=mu, stdvs_matrix=sigma)

        test_for = test_for.reshape([len(test_for), ])
        value = mean_squared_error(means, test_for)
        print("MSE: " + str(value))
        value = pearsonr(means, test_for)
        print("Pearson R: " + str(value))
    # end of for loop

    # save the model
    model.save('model.h5')

    score = model.evaluate(testing_data, testing_range_data)
    print(score)


# end of main 
