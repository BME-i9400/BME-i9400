# BME i9400  
# Fall 2025

# Final Project Information

The main goal of BME i9400 is to prepare you to apply state-of-the-art machine learning algorithms to applications related to your future career. The course’s final project will offer you an opportunity to do exactly this. 

**Projects must be performed individually**

### Important dates
    * Oral presentations (15 minutes): Monday, December 15th, 2025 during class
    * Note: we will likely need to use the Final Exam week to complete the oral presentations  
    * Written report due date: 11:59 PM EST on Friday, December 19th, 2025

## Instructions

In the course project, you are asked to:  

### Part 1 

Choose a dataset and associate learning problem that interests you. Examples include:

    * Predicting life expectancy from activity level and body mass index.  
    * Predicting election results from polling data.  
    * Predicting the occurrence of a seizure from recordings of electrical activity in the brain.

The importance of selecting a problem that interests you cannot be understated: working on a data set that you are passionate about will lead to a better project and prepare you for just the sort of experience that this course is targeting. 

It is critical to explicitly indicate the nature of the feature matrix $X$ and the target variable $y$. 

You must either designate training and test sets (fixed throughout the project), or alternatively, employ some form of cross-validation. 

### Part 2 

Evaluate a "baseline" model -- logistic regression -- to the dataset and report the model performance with the appropriate metrics. How does the performance of the baseline compare to "chance" performance (i.e., if your algorithm was fed mislabeled data)?  

### Part 3

Experiment with more one or more modern learning architectures and attempt to improve the performance of the baseline model:

    * Multilayer Perceptrons
    * Convolutional Networks
    * Sequence models such as Long Short Term Memory (LSTM)
    * Transformers
    * Large Language Models

A key component of the project is to experiment with different parameterizations of the learning algorithm (hyperparamters). You must clearly describe your hyperparameter space and report which set of hyperparameter values produced the best performance. 

### Part 4

Interpret the findings of Parts 2-3 to answer the following questions in narrative form:

    * What was your baseline model's performance on the test set?
        * What does this tell you about the dataset and the problem?
    * What model from Part 3 was able to provide the best performance on the test set? 
        * What does this tell you about the dataset and the problem?
    * Describe the hyperparameter tuning that you performed in Part 3 to arrive at your best model.
    * In the future, what would you attempt to investigate if you wanted to improve performance?

## Written Report

The report should be divided into four sections:

1. _Introduction_: Motivate your problem and explain why you chose your dataset.
2. _Approach_: Describe your implementation, including training/test splits, hyperparameter space, and choice of model architecture.
3. _Results_: provide figures (and tables if necessary) to quantitatively describe the performance of all models tested. 
4. _Discussion_: summary of your inferences and what you learned. Discussion of future work. 

## Grading scheme:

    * Oral presentation: 25%
        * clarity of presentation
        * quality of answers to instructor questions
    * Written report: 75% 
        * how well the problem is motivated 
        * how clearly the findings are described
        * the scope and rigor of model exploration
        * the soundness of inferences drawn

## Additional Information

**Public Datasets**

To carry out the project, you will need to either collect or obtain a data set related to your learning problem. Nowadays, the internet is filled with publicly available data sets on which learning algorithms may be run. Some examples are:

* [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)  
* [https://datasetsearch.research.google.com/](https://datasetsearch.research.google.com/)  
* archive.ics.uci.edu/ml/  
* www.nlm.nih.gov/NIHbmic/nih\_data\_sharing\_repositories.html

You can also ask an LLM to scour the web for a dataset that matches your interests.

**PyTorch Tutorials**

I highly recommend that you take the time to work through one of the two tutorials below. My personal preference is PyTorch Lightning, which abstracts away much of the tedious steps in model training. 

* [​​PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)  
* [PyTorch Lightning tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)


