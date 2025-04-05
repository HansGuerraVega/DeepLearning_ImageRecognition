# Project Description
The supermarket chain Good Seed would like to explore whether data science can help it comply with alcohol laws by ensuring it doesn't sell alcohol to underage people. They're asking you to make that assessment, so when you get started, keep the following in mind:

- Stores are equipped with cameras in the checkout area, which are activated when a person is purchasing alcohol.
- Computer vision methods can be used to determine a person's age from a photo.
- The task, then, is to build and evaluate a model to verify people's ages.
- To begin working on the task, you'll have a set of photographs of people that indicate their ages.

## General Conditions
We're going to use a pre-trained model for the task, specifically one trained on a large image dataset called Imagenet. You can review the content of this dataset on Kaggle: https://www.kaggle.com/c/imagenet-object-localization-challenge/data.

You can find the original data and information for the dataset here: http://www.image-net.org/. Remember to create a Kaggle account if necessary.

This is not strictly a face dataset, so pre-trained weights may not work perfectly for it. However, it is partially valid to use this dataset for face classification because the imagenet dataset is primarily about natural objects and also contains a subset of faces. By setting the 'weights' parameter to 'imagenet', we can load weights from this pre-trained neural network into the ResNet50 architecture.
