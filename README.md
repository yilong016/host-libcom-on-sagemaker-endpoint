# host-libcom-on-sagemaker-endpoint
This project introduce how to host [libcom](https://github.com/bcmi/libcom) on a SageMaker Endpoint and provide testing scenarios by invoking the real-time endpoint. 
We use CloudFormation to build the stack and once that is created, it uses the SageMaker notebooks created in order to create the endpoint and test it.

Create Stack using AWS CloudFormation with [template file]()

<img width="416" alt="image" src="https://github.com/yilong016/host-libcom-on-sagemaker-endpoint/assets/120642887/3852ca35-5f31-43dd-a6b8-28d0aae1f7c1">

Select a unique Stack Name, ackowledge creation of IAM resources, create the stack and wait for a few minutes for it to be successfully deployed

Step1_StackName

Step2_StackIAM

Step3_StackSuccess

go to Cloudformation, find stack name your gave and choose notebook link in the tab resource, click and open it, you will see: deploy.ipynb & predict.ipynb
1. deploy.ipynb: Build you own container including libcom models and prediction algorithm; Create SageMaker endpoint and deploy it, also include Cleanup scripts.
2. Predict.ipynb: Test the deployed endpoint by a serial given sample images.
