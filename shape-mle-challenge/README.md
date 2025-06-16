# Shape's Hard Skill Test - Machine Learning Engineer

## Description of the task
Refactor the `job_test_challenge.py` script into more modularized code, implementing best practices,
writing proper documentation, and making it more suitable for a product release.

## README content
- Description of the strategy and reason behind each implemented change
- Docummentation of planned improvements 

## Changes description
- Include docstrings for better documentation
- Order libraries import in a pythonic manner
- Create a `config.py` file for configuration settings and constants
- Modularize the original methods into classes for making the code more readable, easy to maintain
- Inside the classes, define methods which are task-specific
- Configure logging for proper register and save the occurencies
- Include requirements.txt file

### Details:
- Change from pickle to joblib for loading the model, as joblib is usually faster and more pythonic for sklearn models
- There was no nan values in the input dataset, no need replacing nan by zero

## Planned improvements
- Include pydantic validators for catching errors early, ensure consistency, improve debugging and maintainability, and optimize performance
- Create tests for the main components of the pipeline:
    - parquet dataset loading
    - model loading
    - pipeline building
    - scoring workflow
- Create complementary tests for:
    - edge cases (eg.: very large dataset)
    - error handling
    - integration of all components

*Latest version: June 16th, 2025*

*Solution proposed by Ana Brandão Belisario*
*You can contact her via email at: anabbelisario@gmail.com*
