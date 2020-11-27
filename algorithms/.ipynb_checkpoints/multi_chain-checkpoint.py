import numpy as np

import algorithms.metropolis_hastings as mh


def rmv_burn(thetas, logliks, loglik_tol=1e-1):
    """
    Remove burn-in tail of Markov Chain - to only consider values around sampling distribution.
    
    """
    
    max_loglik, arg_max_loglik = np.max(logliks), np.argmax(logliks)
    max_loglik_theta = thetas[arg_max_loglik]
    
    ls_close = np.isclose(max_loglik, logliks, rtol=0, atol=loglik_tol)
    conv_thetas = thetas[ls_close]

    return (max_loglik_theta, conv_thetas)
    
def gelman_rubin_stat(N, M, theta_means, theta_vars):
    """
    Gelman-Rubin convergence statistic for multiple Markov Chains. Evaluates difference of
    between-chains and within-chain variances for model parameters.
    
    Input:
    
    - N (int): uniform length of Chains (smallest length)
    - M (int): number of Chains
    - theta_means (M x 1 array): sample posterior means for each Chain.
    - theta_vars (M x 1 array): sample posterior variance for each Chain.
    
    Returns;
    
    - R (float): Gelman-Rubin convergence statistic
    
    """
    
    B = N*np.var(theta_means, ddof=1)
    W = np.mean(theta_vars)
    V_hat = ((N-1.0)/N)*W + ((M+1.0)/(M*N))*B
    R = np.sqrt(V_hat/W)
    
    return np.sqrt(V_hat/W)
    
    
def multi_chain(plate_data, plate_corr, star_pos,
                n_steps=20000, width=0.01, init_width=5., n_chain=5,
                sigma_param=False, priors=False):
    """
    Generate multiple Markov Chains to test convergence and increase sample size for inference.
    
    Input:
    
    - plate_data (7 x 2 array): (Dx, Dy) deflection measurements for single plate
    - plate_corr (1 x 2 array): (Dx, Dy) deflection corrections for given plate
    - star_pos (7 x 4 array): (x, y, Ex, Ey) for each star
    - width (float): proposal distribution width
    - init_width (float): proposal distribution width for Chain origin
    - n_steps (int): Number of steps to perform iteration 
    
    Returns:
    
    - R (float): Gelman-Rubin convergence statistic
    """
    # Initialise number of parameters
    if sigma_param:
        N_param = 8
    else:
        N_param = 7
    
    method_kwargs = dict(sigma_param=sigma_param, priors=priors)
    origin = np.zeros(N_param)
    
    all_chains = []
    
    # Generate each Markov Chain with distinct origin 
    for i in range(n_chain):
        param_dict = mh.metropolis_hastings(mh.proposal(origin, init_width, sigma_param),
                                            plate_data, plate_corr, star_pos,
                                            width, n_steps,
                                            method_kwargs)
        
        thetas, logliks = param_dict['thetas'], param_dict['logliks']
        all_chains.append(rmv_burn(thetas, logliks))
    
    chain_lengths = [len(i[1]) for i in all_chains]
    
    short_chain = chain_lenghts.min()
    
    theta_means = [np.mean(i[1][-short_chain:], axis=0) for i in all_chains]
    theta_vars = [np.var(i[1][-short_chain:], axis=0) for i in all_chains]
    
    R = gelman_rubin_stat(short_chain, n_chain, theta_means, theta_vars)
    
    print("{} Chains finalised with convergence R = ".format(n_chain, R))
    return dict(chain_data=all_chains, R=R)