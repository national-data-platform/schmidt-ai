# Schmidt-AI

Welcome to the Schmidt-AI repository! This project is a comprehensive collection of examples for machine learning and data processing on [Nautilus](https://nrp.ai/documentation/). Within this repository, you will find several directories, each containing well-documented use cases that demonstrate practical implementations of various machine learning methods and data processing techniques.

These examples are designed to guide you through the setup, execution, and customization of workflows. Whether you're a beginner looking to understand the basics or an experienced developer aiming to optimize your projects, you'll find valuable resources here to enhance your skills and accelerate your development process.


## minist-pytorch Usage Instructions

Below are the steps to run minist-pytorch using different environments. [main.py](/mnist-pytorch/main.py) was taken from the [official pytorch examples](https://github.com/pytorch/examples/blob/main/mnist/main.py) and adapted for this tutorial.

Note you will have to `cd` into [mnist-pytorch](/mnist-pytorch/) for the rest of the README (except for Kubernetes).
```
cd mnist-pytorch
```

To further modify the environment (if needed), create .env with the following contents:
```
S3_ENDPOINT=https://s3-west.nrp-nautilus.io

```

Note that the mnist dataset was uploaded to a public [NRP s3 bucket](https://nrp.ai/documentation/userdocs/storage/ceph-s3/) already so the rest of these steps are not needed and are only meant for completeness.

Download mnist dataset locally:
```
python3 download-mnist.py
```

Upload dataset to schmidt-ai bucket and make it public:
```
aws s3 cp ./data/MNIST/raw s3://schmidt-ai/mnist/ --recursive --profile nrp --endpoint-url https://s3-west.nrp-nautilus.io --acl public-read
```

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
3. **Run the Docker Container with the .env File:**
  - Use the `--env-file` option to load the environment variables from the `.env` file when running the container:
    ```
    docker run -it --rm --env-file .env minist-pytorch
    ```

## Using Kubernetes ([NRP](https://nrp.ai/documentation/))
For using Kubernetes, there are already images built to use the [job manifest](/kubernetes/mnist-pytorch/job-s3.yaml) in [NRPs gitlab registry](https://gitlab.nrp-nautilus.io/ndp/schmidt-ai/container_registry/5024).

Why Use Kubernetes for ML Training Jobs on NRP:
- **Scalability:**  
  Kubernetes allows you to easily scale your training jobs by dynamically adjusting the number of pods based on workload, ensuring efficient resource usage.

- **Resource Management:**  
  It provides robust scheduling and resource allocation, making it simple to manage GPU and CPU resources for demanding machine learning tasks.

- **Fault Tolerance:**  
  Automatic restarts and pod health checks help maintain job reliability, minimizing downtime and ensuring training jobs complete successfully.

- **Environment Consistency:**  
  Containers guarantee that your ML environments remain consistent across different stages, reducing configuration errors and streamlining deployments.

With your `.env` file ready, create a Kubernetes secret:
```
kubectl create secret generic mnist-pytorch --from-env-file=.env
```
This command packages your environment variables into a secret named `mnist-pytorch`.

Note that if you are using the s3 bucket for the mnist dataset, the s3 endpoint must be changed to the inside endpoint:
```
S3_ENDPOINT=http://rook-ceph-rgw-nautiluss3.rook

```

Create a job using data that is stored in a s3 bucket:
```
kubectl create -f kubernetes/mnist-pytorch/job-s3.yaml
```

Create a job using data that is stored in a [pvc](https://nrp.ai/documentation/userdocs/tutorial/storage/#creating-a-persistent-volume-claim):
```
kubectl create -f kubernetes/mnist-pytroch/pvc.yaml
kubectl create -f kubernetes/mnist-pytorch/job-s3.yaml
```
Note that creating a pvc is only needed once. 

View your pod statuses by executing:
```
kubectl get pods
```
For detailed logs on a specific pod, use:
```
kubectl logs <pod-name>
```

To remove the job (and its associated pods), run:
```
kubectl delete job minist-pytorch-job
```
Note that this is not needed and can just let the job clean up itself (set to 10 minutes after all pods have compelted)