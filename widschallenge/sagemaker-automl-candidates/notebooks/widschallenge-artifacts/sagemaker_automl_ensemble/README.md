# Amazon SageMaker Autopilot AutoML Notebook Helper Library

This package contains companion helper code that will help you abstract the complexity of low-level 
interaction with the Amazon SageMaker Python SDK for the Autopilot interactive workflow notebook.

## Workflow and features overview 

`AutoMLLocalEnsembleRunConfig`

This is a configuration class that keeps track of all input and output to Amazon Simple Storage Service 
(Amazon S3) paths, conventions, and AWS and Amazon SageMaker shared variables for an interactive execution of Autopilot's Ensemble mode trials.

## Requirements

The library is compatible with python 3.6+ and tested with IPython 6.4.0.
