#!/usr/bin/env python
# coding: utf-8

# # Frequentist vs. Bayesian Statistics

# ### Load Necessary Modules

import numpy as np
from scipy.stats import beta as beta
from scipy.stats import binom_test
import matplotlib.pyplot as plt
from math import factorial
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ## Create functions for plotting distribution


def from_mu_sigma(mu, sigma):
    a = ((1-mu)/sigma**2 - 1/mu)*mu**2
    b = a*(1/mu - 1)
    return a,b

def plot_beta_distribution(ax, a=None, b=None, mu=None, sigma=None, label=None, color='blue'):
    if a == None and mu != None:
        a,b = from_mu_sigma(mu, sigma)
    x = np.linspace(0.0, 1.0, 1000)
    ax.plot(x, beta.pdf(x, a, b),color=color, lw=3, alpha=0.6, label=label)
    ax.fill_between(x, beta.pdf(x, a, b), alpha=0.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0,beta.pdf(x, a, b).max()*1.05)
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.axvline(0.5, color='k', lw=2)
    ax.legend(prop={'size':14})
    return


# ## Get probability of fairness equation for Bayesian Stats


def get_prob_fair(a, b):
    x_test = np.linspace(0.45, 0.55, 1000)
    prob_fair = np.trapz(beta.pdf(x_test, a, b), x=x_test)
    return prob_fair


# ## plot beta posterior based on beta prior

def posterior(heads, tails, mu, sigma, ax, kwargs={}):
    N = heads + tails
    z = heads
    a,b = from_mu_sigma(mu, sigma)
    post_a = z+a
    post_b = N-z+b
    
    kwargs['label'] = kwargs['label'] + ' %i H, %i T'%(heads, tails)
    
    plot_beta_distribution(ax, a=post_a, b=post_b, **kwargs)
    x_test = np.linspace(0.45, 0.55, 1000)
    prob_fair = np.trapz(beta.pdf(x_test, post_a, post_b), x=x_test)
    p_val = binom_test([heads, tails])
    print('p:', 1-p_val)
    print("%i Heads, %i Tails --> Probability Coin is Fair: %.3g"%(heads, tails, prob_fair))
    return


def make_plot(h_t_list):
    for H, T in h_t_list:
        fig = plt.figure()
        ax = plt.gca()
        plot_beta_distribution(ax, a=H, b=T, **{'color':'blue'})
        prob_fair = get_prob_fair(H, T)
        ax.set_title('H: %i  T:%i  P(fair): %.2g'%(H,T, prob_fair), fontsize=20)

def make_real_flip_plot(h_t_list, prior_mean, prior_standard_deviation):
    for H, T in h_t_list:

        N = H + T
        z = H
        a,b = from_mu_sigma(prior_mean, prior_standard_deviation)
        post_a = z+a
        post_b = N-z+b

        fig = plt.figure()
        fig.set_size_inches(8.0, 5.0)
        ax = plt.gca()
        plot_beta_distribution(ax, mu=prior_mean, sigma=prior_standard_deviation, **{'color':'blue', 'label':'prior'})
        prob_fair = get_prob_fair(post_a, post_b)
        ax.set_title('H: %i  T:%i  P(fair): %.2g'%(H,T, prob_fair), fontsize=20)
    
        posterior(H,T, prior_mean, prior_standard_deviation, ax, kwargs={'color':'red', 'label':'posterior'})


