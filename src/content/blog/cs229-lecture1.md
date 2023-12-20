---
title: Notes for CS229 - Lecture 1 & 2
description:
  Notes for Stanford's CS229 Machine Learning Module
pubDatetime: 2022-12-11T15:22:00Z
postSlug: cs229-lecture1
featured: true
draft: false
tags:
  - cs229
  - ai
  - ml
---

Recently, I have embarked on the journey of learning machine learning. The CS229 module was recommended indirectly on the Lex Fridman podcast with Andrej Karpathy (Tesla Vision/OpenAI). Karpathy taught the module CS231n (which is a renowned course on vision/deep learning) and the CS229 module was a pre-requisite to the CS231n module, and so I begin my journey on picking up the necessary skills to better understand deep learning.

I have minified my notes of CS229 Lecture's 1 and 2 below, for full context and detail see the original notes [here](https://cs229.stanford.edu/notes2022fall/main_notes.pdf). I may also add useful context from non-course material in my explanations.

## Table of contents

## CS229 course overview
The course topics can be summarised as follows:

1. Supervised Learning
2. Deep Learning
3. Generalisation and Regularisation
4. Unsupervised Learning
5. Reinforcement Learning and Control

## Supervised Learning
### Definition
>Supervised learning (SL) is a paradigm in machine learning where input objects and a desired output value train a model. The training data is processed, building a function that maps new data on expected output values. An optimal scenario will allow for the algorithm to correctly determine output values for unseen instances. -
[paraphrased from Wikipedia](https://en.wikipedia.org/wiki/Supervised_learning)

### Example
Given the data below, can we predict an unseen house's price?

| Square footage | #bedrooms | Price (£1000s) |
|----------------|------------|----------------|
| 300            | 4          | 450            |
| 200            | 2          | 300            |
| 100            | 1          | 120            |

Here, we can introduce some terminology:

- $x^i$, the input variables or **features**

- $y^i$, the output or target variable

- $(x^i, y^i)$ a **training example**

- $[(x^i, y^i), ...]$ a list of training examples = a **training set**

Our goal, is given a training set, to learn a function $h(x) = y$, so that
$h(x)$ is a "good" predictor for the corresponding $y$ value.

- When the target variable, $y$, is continuous (e.g. in the housing example), this is called a **regression** problem
- When $y$ can take on a small number of discrete values
(e.g. "is it a house or an apartment") we call this a **classification** problem
 

## Linear regression
Linear regression is one type of supervised machine-learning algorithm. It works by first approximating $y$ as a linear function of $x$. Our approximation or hypothesis of $y$ can be defined by:

$h_{\theta}(x) = {\theta}_0 + {\theta}_1x_1 + {\theta}_2x_2$

$\theta_i$'s are the parameters, also called **weights**. We are aiming to learn these values 
through the algorithms discussed here on.

If we set $x_0 = 1$, the above equation can be simplified to:

$h(x) = \displaystyle\sum_{i=0}^{d} \theta_{i}x_{i} = \theta^T{x}$, where $\theta^T$ and $x$ are vectors (so the last part is the dot product.) $d$ is the number of input variables (not counting $x_0$).

Given a training set, how can we learn the parameters $\theta$? One method seems to be to make $h(x)$ as close to $y$, at least for the training examples we have available.

### Cost function
To be able get $h(x)$ as close to $y$ as possible we need a way to measure the closeness through least means squared (LMS):

$J(\theta) = \displaystyle\frac{1}{2} \displaystyle\sum_{i=1}^{n}(h_{\theta}(x^{(i)}) - y^{(i)})^2$

We will show later on why LMS it is thought to be the best method.

### Gradient descent

We now want to choose $\theta$ to minimise $J(\theta)$. The algorithm for doing this will involve: choosing an initial guess of $\theta$, repeatedly change $\theta$ to make $J(\theta)$ smaller, until hopefully we converge to a value of $\theta$ that minimises $J(\theta)$.

This can be shown in the gradient descent algorithm:

$\theta_{j} := \theta_{j} - \alpha \displaystyle\frac{\partial}{\partial\theta_{j}}J(\theta)$

- $:=$ means "update to"
- $\alpha$ is the learning rate

### The normal equations
Alternative to gradient descent. Closed form version (able to work out in one step)

$\theta = (X^{T}X)^{-1}X^{T}\vec{y}.^{3}$

## Locally Weighted Regression (LWR)
A straight line isn't always the line of best fit. We can use a quadratic function.

Do we want x^2 or sqrt(x) or log(x), in a later lecture we will discuss feature selection.

In ML we distinguish between parametric and non parametric learning algorithms.

Parametric learning algorithm: fit fixed set of parameters to data
Non-parametric learning algorithm: Amount of data/parameters you need to keep grows (linearly)
with size of data

Locally weighted is different to linear regression in that you look at the data around the
point you want to make a prediction, and then you make a straight line / prediction around these close by points.

### Modified cost function
Fit $\theta$ to minimise
$\displaystyle\sum_{i=1}^{m} w^{(i)}(y^{(i)}-\theta^{T}x^{(i)})^2$
We have simply added $w$. Where $w^{(i)}$ is a "weighting" function.
$w^{(i)} = \exp(-\frac{(x^{(i)}-x)^2}{2\tau^2})$

- This means if $abs(x^{(i)} - x)$ is small, $w^{(i)} \approx 1$
- This means if $abs(x^{(i)} - x)$ is large, $w^{(i)} \approx 0$

Explained in plain English: if an example x^i is far from where you want to make a prediction, multiply by 0 or something close to 0. if it's close, then multiply by 1. So the net effect it sums over only the values of x close to where you want to make a prediction.

The w function plotted is shaped similarly to a gaussian density (normal distribution), bell curve, where the bell curve is the weight, w, assigned to the training data point, and so nearby ones are weighted higher up the bell, and outer ones are near 0.

How wide should you choose the gaussian density? We call it bandwith parameter, $\tau$. The choice of this parameter has an effect on over or underfitting. Good to try this practically with code, try a varying $\tau$/.

Andrew recommends using LWR when there is a relatively low dimensional data (low number of features, n is quite small) and you have a lot data and you don't want to think about what features to use.

## Probabilistic interpretation of Linear Regression

Why least squares? Why not $()^4$ or something else?

Assume a house's price is a linear function of the living space and it's number of bedrooms plus the error term, $\epsilon$, which captures unmodelled effects, random noise (such as the seller is an unusually good or bad mood or it's near a school district, which makes the price go up or down) $y^{(i)} = \theta^{T}x^{(i)} + \epsilon^{(i)}$

We assume the error is Gaussian distributed and IID (indepedently, and identicaly distributed from each other.)

If we perform maximum likelihood estimation (see lecture notes for full working out) it turns out to be the exact same equation as the cost function $J(\theta)$, so this is one method of proof for the least squares cost function finding the correct values of $\theta$.

Mathematically, maximizing the likelihood function helps us obtain parameter estimates that are consistent with the observed data, making it a key concept in statistical inference and estimation.

The reason we have looked at this is it's giving us a framework to our first classification problem:
1) Make assumptions of P(Y | X; 0), Gaussian errors and IID
2) Figure out maximum likelihood estimation (in this case we derived an algorithm which turned out to be the least squares algorithm)
