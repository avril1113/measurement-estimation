import numpy as np
from keras import backend as K
from keras.layers import Dense, concatenate


def add_univariate_mixture_layer(previous_layer, nb_components):
    """
    This functions adds a univariate mixture layer onto previous_layer
    """

    # layers
    layer1 = Dense(nb_components, activation=K.softmax)(previous_layer)
    layer2 = Dense(nb_components)(previous_layer)
    layer3 = Dense(nb_components, activation=K.exp)(previous_layer)

    output_layer = concatenate([layer1, layer2, layer3])

    # Return the concatenated output layer
    return output_layer

# end def


def separate_mixture_matrix_into_parameters(mdn_output_matrix, nb_components):
    """
    This function takes the output matrix of a mixture density network and separates 
    the output matrix into three separate individual matrices: the mixture coefficient 
    matrix, the means matrix, and the standard deviations matrix.
    """

    pi = mdn_output_matrix[:, :nb_components]
    mu = mdn_output_matrix[:, nb_components:2 * nb_components]
    sigma = mdn_output_matrix[:, 2 * nb_components:]

    # Return statements
    return pi, mu, sigma
    
# end def


def compute_gaussian_kernel_probability_matrix(target_array, means_matrix, stdvs_matrix):
    """
    This functions computes the probability of each Gaussian kernel using the means_matrix 
    the stdvs_matrix.
    """
    # Convert the 'target' array into a row vector
    target_matrix = K.reshape(target_array, [K.shape(target_array)[0], 1])

    # 'Tile' the target row vector to match the width of the means matrix
    target_matrix = K.tile(target_matrix, [1, K.shape(means_matrix)[1]])
    
    # Compute the exponential portion of the Gaussian function
    result = target_matrix - means_matrix
    result = result * (1 / (stdvs_matrix + 1e-10))
    result = -K.square(result) / 2

    # Compute the Gaussian function's normalizer
    one_div_sqrt_two_pi = 1 / np.sqrt(np.multiply(2, np.pi))
    result = K.exp(result) * (1 / (stdvs_matrix + 1e-10))
    result = result * one_div_sqrt_two_pi

    # Return the result of the product
    return result

# end def


def compute_total_probability_vector(mix_coeff_matrix, kernel_probability_matrix):
    """
    Computes the total, weighted probability vector using the mixture coefficient matrix 
    and the kernel probability matrix.
    """
    probability = K.sum(mix_coeff_matrix * kernel_probability_matrix, axis=1, keepdims=True)
    
    # Return statement
    return probability

# end def


def negative_log_likelihood_loss(nb_components, space_holder_param=None):
    """
    This function serves as a wrapper for the negative log likelihood loss function.
    """

    def loss_fnc(target_groundtruth, target_predicted):
        """
        Computes the negative log likelihood loss.
        """

        mix_coeffs_matrix, means_matrix, stdvs_matrix = separate_mixture_matrix_into_parameters(target_predicted, nb_components)
        loss = compute_gaussian_kernel_probability_matrix(target_groundtruth, means_matrix, stdvs_matrix)
        loss = compute_total_probability_vector(loss, mix_coeffs_matrix)
        loss = K.sum(loss, axis=1, keepdims=True)
        loss = -K.log(loss + 1e-10)
        loss = K.sum(loss)

        # Return the log loss
        return loss

    # end def
    return loss_fnc
# end def


def compute_mixture_total_mean_variance(mix_coeff_matrix, means_matrix, stdvs_matrix):
    """
    Computes the total mean and the total variance of the entire distribution given a set of 
    mixture coefficients, means, and standard deviations.
    """
    mean = np.multiply(mix_coeff_matrix, means_matrix)
    mean_kd = np.sum(mean, axis=1, keepdims=True)
    mean = np.sum(mean, axis=1)
    variance = np.subtract(means_matrix, mean_kd)
    variance = np.square(variance)
    variance = np.add(np.square(stdvs_matrix), variance)
    variance = np.multiply(mix_coeff_matrix, variance)
    variance = np.sum(variance, axis=1)
    variance = np.ravel(variance)

    # Return the total mean and total variance as a tuple
    return mean, variance

# end def


def compute_max_component_mean_variance(mix_coeff_matrix, means_matrix, stdvs_matrix):
    """
    Returns the mean and standard deviation from the component with the highest mixture 
    coefficient.
    """

    # Find the component with the largest mixture coefficient
    max_index = np.argmax(mix_coeff_matrix, axis=1)

    # Compute the mean and standard deviation
    mean = np.empty(len(max_index))
    stdv = np.empty(len(max_index))

    for n in range(len(max_index)):
        mean[n] = means_matrix[n, max_index[n]]
        stdv[n] = stdvs_matrix[n, max_index[n]]

    # Return the computed mean and standard deviation
    return mean, stdv

# end def
