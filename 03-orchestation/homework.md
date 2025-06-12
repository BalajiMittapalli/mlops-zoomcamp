# MLOps Zoomcamp - Homework 3: Orchestration

## Question 1. Select the Tool

What's the name of the orchestrator you chose?

* Apache Airflow

## Question 2. Version

What's the version of the orchestrator?

* 3.0.1


## Question 3. Creating a pipeline

Let's read the March 2023 Yellow taxi trips data.

How many records did we load?

* 3,403,766

![image1](https://github.com/user-attachments/assets/19255fcc-abf5-452a-959c-7ce2c9543d34)


## Question 4. Data preparation


Let's apply to the data we loaded in question 3.

What's the size of the result?

* 3,316,216

![image2](https://github.com/user-attachments/assets/480b5625-3d9f-4d9e-a278-135008b763b5)


## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1.

Fit a dict vectorizer.
Train a linear regression with default parameters.
Use pick up and drop off locations separately, don't create a combination feature.
Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model.

What's the intercept of the model?

Hint: print the intercept_ field in the code block

* 24.77(23.82889069001763)

![image3](https://github.com/user-attachments/assets/be1286c4-cd01-4b55-9072-fcd81a7a0be1)

![mlflow_parameters](https://github.com/user-attachments/assets/dd50bb6e-0be2-4f46-99b8-dd01c573f911)



## Question 6. Register the model

The model is trained, so let's save it with MLFlow.

Find the logged model, and find MLModel file. What's the size of the model? (model_size_bytes field):

* 9,534
  
![model_size](https://github.com/user-attachments/assets/b69bcca6-01f6-490e-9d70-5344453f889c)


