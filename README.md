# Schmidt-AI

Welcome to the Schmidt-AI repository! This project is a comprehensive collection of examples for machine learning and data processing on [Nautilus](https://nrp.ai/documentation/). Within this repository, you will find several directories, each containing well-documented use cases that demonstrate practical implementations of various machine learning methods and data processing techniques.

These examples are designed to guide you through the setup, execution, and customization of workflows. Whether you're a beginner looking to understand the basics or an experienced developer aiming to optimize your projects, you'll find valuable resources here to enhance your skills and accelerate your development process.


## minist-pytorch Usage Instructions

Below are the steps to run minist-pytorch using different environments. [main.py](/mnist-pytorch/main.py) was taken from the [official pytorch examples](https://github.com/pytorch/examples/blob/main/mnist/main.py) and adapted for this tutorial.

## Using Pip Environment

1. **Set Up a Virtual Environment:**
    - With Python's built-in venv:
      ```
      python3 -m venv env
      source env/bin/activate   # On Windows use: env\Scripts\activate
      ```
2. **Install Dependencies:**
    - Install PyTorch, torchvision, and any other required packages:
      ```
      pip3 install -r requirements.txt
      ```
3. **Run the Application:**
    - Once dependencies are installed, run:
      ```
      python main.py
      ```
4. **Deleting a Pip Virtual Environment:**

    1. **Deactivate the Environment (if active):**
        - Simply run:
          ```
          deactivate
          ```
    2. **Remove the Virtual Environment Directory:**
        - **On macOS/Linux:**
          ```
          rm -rf env
          ```
        - **On Windows:**
          ```
          rmdir /s /q env
          ```

    Ensure you're not inside the virtual environment directory when deleting it.

## Using Docker

1. **Build the Docker Image:**
    ```
    docker build -t minist-pytorch .
    ```
2. **Run the Docker Container:**
    ```
    docker run -it --rm minist-pytorch
    ```

Follow these instructions to run minist-pytorch in your desired environment.