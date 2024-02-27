---
title: Notes for CS229 - Lecture 4
description:
  Minified notes for Stanford's CS229 Machine Learning Module
pubDatetime: 2022-12-14T15:54:00Z
postSlug: cs229-lecture4
featured: true
draft: false
tags:
  - cs229
  - ai
  - ml
---

# Perceptron & Generalized Linear Model

## Table of contents

## Perceptron

The Perceptron is not widely used in practice, but it's a simple concept to understand and is historical for the field.

Logistic Regression uses the sigmoid function. Perceptron uses a slightly different function, where: $g(z) =$
1 if z >= 0, 0 if z < 0

If we then let $h_{\theta}(x) = g(\theta^{T}x)$ as before, but now we've slightly changed
$g$, and if we use the gradient descent update rule (same as before):

$\theta_{j} := \theta_{j} + \alpha(y^{(i)} - h_{\theta}(x^{(i)}))x^{(i)}_j$

Again, it looks the same, but we have modified $g$, this is the **pereptron learning algorithm**.

## Generalized Linear Models (GLMs)

GLMs are a class of statistical models that extend linear regression to accomodate non-Gaussian distributions and non-constant variance.

Linear and logistic regression belong to the family of GLMs.

### The exponential family

GLMs are designed to handle a broad class of distributions within the exponential family.

**Definition:** a class of distributions is in the exponential family if it can be written in the form:

$p(y; \eta) = b(y)\exp(\eta^{T}T(y) - a(\eta))$

The lecture notes then show that Bernoulli and the Gaussian distributions are examples of exponential family distributions, as they can be rearranged into the form above.

There's other distributions that are members of the exponential family: e.g. binomial, Poisson (for modelling count-data), the gamma and the exponential (for modelling continuous, non-negative random variables, such as time-intervals), the beta and the Dirichlet (for distributions over probabilities), and many more.

### Constructing GLMs
Consider a classification or regression problem where we would like to predict the value of some random variable $y$ as a function of $x$

To derive a GLM we will make the following three assumptions:

1. $y | x;\theta ~ ExponentialFamily(\eta)$ the distribution of
$y$ follows some exponential family distribution, with parameter $\eta$

### Softmax regression

Another example of a GLM used for multi-class classification

Each class has it's own set of parameters. This is a generalization of logistic regression.
(Show example image of spliting up circles, triangles, squares on graph).

In soft max we put to power of e exp(), so we have positive values. Then we normalise, divide by the sum of all of them, so it outputs a probability distribution as opposed to a scalar.
