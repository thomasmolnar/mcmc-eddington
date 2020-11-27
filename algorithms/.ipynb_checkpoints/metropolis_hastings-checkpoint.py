import numpy as np
import random

def log_likelihood(theta, plate_data, star_pos,
                   sigma_param=False, priors=False):
    """
    Compute log-likelihood given set of theoretical (GR) delfection parameters
    for a single plate. 
    
    Input:
    
    - theta (7 x 1 numpy array): [a, b, c, alpha, d, e, f (, sigma)]
    input parameters for model
    - plate_data (7 x 2 array): (Dx, Dy) deflection measurements for single plate
    - star_pos (7 x 4 array): (x, y, Ex, Ey) for each star
    - sigma_param (bool): condition to consider sigma a parameter or not 
    - priors (bool): condition to add strong priors on parameters
    
    Returns:
    
    - likelihood (float): log likelihood of model given input parameters
    
    """
    # Theoretical Delfections:
    # Dx = ax + by + c + alpha*Ex, Dy = dx + ey + f + alpha*Ey
    theory_Dx = np.array([theta[0]*i[0]+theta[1]*i[1]+theta[2]+theta[3]*i[2] for i in star_pos])
    theory_Dy = np.array([theta[4]*i[0]+theta[5]*i[1]+theta[6]+theta[3]*i[3] for i in star_pos])

    # Measured Deflections (corrected):
    mes_Dx = plate_data[:, 0]*6.25 # 6.25 unit plate units 
    mes_Dy = plate_data[:, 1]*6.25

    assert len(theory_Dx)==len(mes_Dx) and len(theory_Dy)==len(mes_Dy)
    
    if sigma_param:
        sigma = parameters[7]
    else:
        sigma = 0.05 # Assumed measurement error in arcsecs
        
    # Now calculate log likelihood
    const = 1./(((np.sqrt(2*np.pi))*sigma)**(len(theory_Dx) + len(theory_Dx)))  
    exp_x = (((mes_Dx - theory_Dx)**2)/(2*(sigma)**2))  
    exp_y = (((mes_Dy - theory_Dy)**2)/(2*(sigma)**2))
    
    log_lik = -np.sum(exp_x) - np.sum(exp_y) + np.log(const)
    
    # Add strong priors for nuisance parameters centered around 0 
    if priors and not sigma:
        for i in range(len(theta)):
            log_lik -= (theta[i]**2)/(2*(sigma)**2)
     
    return log_lik


def proposal(theta, width, sigma_param=False):      
    """
    Returns new parameters via proposal Gaussian distribution centred around
    current parameters.
    
    Input:
    
    - theta (numpy array): input parameters
    - width (float): proposal width of Gaussian distribution
    - sigma_param (bool): condition for sigma to be a parameter to be inferred aswell
    
    Return:
    
    - new_theta (numpy array): proposal parameters
    """
    
    new_theta = [np.float64(i) + random.gauss(0,width) for i in theta]
    if sigma_param:
        new_theta[7] = np.abs(theta[7] + random.gauss(0,(width)/5))
    
    return new_theta


def metropolis_hastings(init_theta, plate_data, star_pos,
                        width=0.01, n_steps=20000,
                        method_kwargs = dict(sigma_param=False, priors=False)):
    """
    Metropolis-Hastings iterator for single plate. 
    
    Input:
    
    - init_theta (7 x 1 numpy array): Origin in hyperparameter space
    - plate_data (7 x 2 array): (Dx, Dy) deflection measurements for single plate
    - star_pos (7 x 4 array): (x, y, Ex, Ey) for each star
    - width (float): proposal distribution width
    - n_steps (int): Number of steps to perform iteration 
    
    Returns:
    
    - {'thetas', 'logliks'}: parameters and log likelihood values of all points in Markov Chain
    
    """
    print("Loading Monte Carlo Monte Chain of {} steps..".format(n_steps))
    theta_chain, loglik_values = [] , []
    theta_chain.append(init_theta)    
    loglik_values.append(log_likelihood(init_theta, plate_data=plate_data, star_pos=star_pos,
                                   **method_kwargs))
    curr_theta = init_theta # Initialise Markov Chain
    
    for i in range(n_steps):
        curr_loglik = log_likelihood(curr_theta, plate_data, star_pos,
                                   **method_kwargs)  
        # Proposed successive point in Markov Chain 
        prop_theta = proposal(init_theta, width, sigma_param=method_kwargs['sigma_param'])        
        prop_loglik = log_likelihood(prop_theta, plate_data=plate_data, star_pos=star_pos,
                                   **method_kwargs)
        
        alpha = prop_loglik - curr_loglik                    
        
        if np.log(random.random()) < alpha:
            curr_theta = prop_theta
            theta_chain.append(prop_theta)  
            loglik_values.append(prop_loglik)    
            
        else:
            pass
    
    print('Monte Carlo Markov Chain loaded, most likeli theta = {}'.format(theta_chain[np.argmax(loglik_values)]))
    return dict(thetas=np.array(theta_chain), logliks=np.array(loglik_values))