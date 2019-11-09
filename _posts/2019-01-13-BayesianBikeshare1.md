---
layout: post
comments: true
title:  "Bayesian Modeling for Ford GoBike Ridership with PyMC3"
excerpt: "Part I: Linear Models"
date:   2019-01-13
---

This post originally appeared on Towards Data Science [here]("https://towardsdatascience.com/bayesian-modeling-for-ford-gobike-ridership-with-pymc3-part-i-b905104af0df"), but 
exists behind a paywall.

Bike shares are a large part of the transport equation for cities around the world. In San Francisco, one of the major players in the bike share game is Ford with its GoBike program. Conveniently, they kindly release their data for people like me to study. I wonder if it is possible to easily forecast the ridership of the next day in order to ensure enough bikes are available to riders, based on past information?
This would be a fairly trivial task to complete if I were to use sklearn to build a Linear Regression model. Often I find myself looking for data sets to learn a new tool or skill in Machine Learning. I have been trying to find an excuse to try one of the probabilistic programming packages (like [PyStan]("https://pystan.readthedocs.io/en/latest/") or [PyMC3]("https://docs.pymc.io/")) for years now, and this bike share data seemed like a great fit.
A number of companies are looking towards Bayesian inference for their internal predictive models. As computational cost goes down, the notoriously long training time for these systems decreases. Notably Uber released [Pyro]("https://eng.uber.com/pyro/"), an open source framework that can appears to be fairly flexible and easy to use. Quantopian is a frequent user of PyMC3. Booz Allen Hamilton and GoDaddy are two other firms that I am aware of that are pursuing these types of ML models.

In this post I have constructed a simple example that I hope will be instructive to a PyMC3 beginner. I downloaded GoBike data from 2017 — September 2018. I then aggregated the data on a day-by-day basis to assemble information about the ridership details (e.g., mean age, subscription membership), the length of their rides, and the number of total riders.
The issue with these Bayesian ML tools is that they can take a long time to train, especially on massive data. For this aggregated dataset, I have only 100s of rows, which is easily trainable on a laptop. After a little research, I settled on learning PyMC3 as my package of choice. It seems like it has a number of great tutorials, a vibrant community, and a fairly easy to use framework.

<div class="imgcap">
<img src="https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/1.png">
</div>

![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/1.png "image")

I begin this illustrative example by scaling the data with a RobustScaler and plotting the seaborn correlation heatmap to see if there are any patterns our model could learn. We are attempting to predict the value of nextDay. From the heatmap, we can see some simple relationships between variables,
both positive (scaled_total_riders) and negative (scaled_duration_hrs).
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/2.png "image")

Let us make a naive prediction and use that as a baseline. What would the RMSE be if we just took the average number of daily riders and used that as our prediction? We would expect to be off by approximately 1900 riders per day by taking this easy approach.
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/3.png "image")

A linear model with sklearn performs slightly better in RMSE, and is quite easy to implement. The model is a series of weights for each variable in our data, in addition to an intercept. How do we interpret the confidence our model has in each of those individual parameters?
This can be where the Bayesian magic really shines. It undoubtedly takes more lines of code, more thought, and longer training time than the sklearn example above. I promise you, dear reader, that it can all be worth it.
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/4.png "image")
First, PyMC3 runs on Theano under the hood. We have to make some slight changes to our pandas/numpy data, and the most major change is by setting a shared tensor, as follows.
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/5.png "image")

When we look to make predictions for our model, we can swap out X_train for X_test and use the same variable name.
Now that we have our data set up, we need to build our model, which we initialize by calling pm.Model() . Inside of this model context, we need to build our complete set of assumption about our priors (parameters) and output. Normal (Gaussian) distributions are fairly safe bets for your first model.
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/6.png "image")
This constitutes our model specification. Now we have to learn what the posterior distribution of our model weights could be. Unlike sklearn, the coefficients are now a distribution of values, not a single point. We sample a range of possible weights, and the coefficients that appear to fit our data well are retained in something called a trace. The sampling functions (NUTS, Metropolis, et al.) are well beyond the scope of this post, but there are vast repositories of knowledge describing them. Here we build our trace from our model:
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/7.png "image")
The NUTS sampler complains that using find_MAP() is not a good idea, but frequently this is used in [tutorials]("https://people.duke.edu/~ccc14/sta-663/PyMC3.html"), and did not seem to hurt my performance.
We can also try a different sampler that tries to approximate the posterior distributions:
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/8.png "image")

The trace can be plotted, and generally looks like this. The beta parameters look fairly constrained in distribution (left plots) and seem to be reasonably consistent across the last 1000 sampled items in our trace (right plot). The alpha parameter looks less certain.
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/9.png "image")
Now that we have our posterior samples, we can make some predictions. We generally observe a so-called ‘burn in’ period in PyMC3 where we discard the first thousand samples of our trace (trace[1000:]), as these values may not have converged. We than draw 1000 sample weights from this trace, calculate what the predictions might be, and take the mean of that value as our most probable prediction for that data point. From here, we simply calculate the RMSE.
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/10.png "image")
If we want to test on our holdout dataset :
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/11.png "image")
So this model that we built performs better than our naive approach (average ridership) but slightly worse than our sklearn model. In an included example in the Github repo, I was able to build a similar model that beat the sklearn model by scaling the Y value, and modeling it as a Normally distributed variable.
Further tuning of the model parameters, using different scalings, assuming a wider range of possible beta parameters can all be employed to lower the RMSE of this example. The goal of this post is to introduce the basics of model building and provide an editable example that you can play around with and learn from! I encourage you to provide feedback in the comments section below.
To recap, there is a price to pay for Bayesian models. It certainly takes longer to implement and write a model. It requires some background knowledge on Bayesian statistics. The training time is orders of magnitude longer than using sklearn. However, tools like PyMC3 can offer greater control, understanding, and appreciation for your data and the model artifacts.
Although there are a number of good tutorials in PyMC3 (including its documentation page) the best resource I found was a [video]("https://www.youtube.com/watch?v=rZvro4-nFIk") by Nicole Carlson. It explores how a sklearn-familiar data scientist would build a PyMC3 model. Careful readers will find numerous examples that I adopted from that video. I also learned a lot from [Probabilistic Programming and Bayesian Methods for Hackers]("https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Prologue/Prologue.ipynb"), which is a free notebook based tutorial on practical Bayesian models using PyMC3. These two resources are absolutely amazing. Duke also has an example website that has numerous data situations that I found informative. Towards Data Science has also hosted a number of cool posts throughout the year that focused on Bayesian analysis and have helped inspire this post.
In the next blog post, I will illustrate how to build a Hierarchical Linear Model (HLM) that will greatly improve the performance of our initial approach. Below are a Kaggle kernel that you can fork and a Github repo that you can clone to play around with the data and develop your own PyMC3 models with. Thank you for reading!
![alt text](https://github.com/franckjay/franckjay.github.io/tree/master/_assets/BayesianBikeshare/12.png "image")

[DayByDayPredictions | Kaggle](https://www.kaggle.com/franckjay/daybydaypredictions?source=post_page-----b905104af0df----------------------)

[Github Repo](https://github.com/franckjay/FordGoBike)