# host-libcom-on-sagemaker-endpoint
This project introduce how to host libcom on a SageMaker Endpoint and provide testing scenarios by invoking the real-time endpoint. 
We use CloudFormation to build the stack and once that is created, it uses the SageMaker notebooks created in order to create the endpoint and test it.

Create Stack using AWS CloudFormation:

Choose Launch Stack and (if prompted) log into your AWS account: Launch Stack
Select a unique Stack Name, ackowledge creation of IAM resources, create the stack and wait for a few minutes for it to be successfully deployed

Step1_StackName

Step2_StackIAM

Step3_StackSuccess

go to Cloudformation, find stack name your gave and choose notebook link in the tab resource, click and open it, you will see: deploy.ipynb & predictor.ipynb
1. deploy.ipynb: Build you own container including libcom models and prediction algorithm; Create SageMaker endpoint and deploy it, also include Cleanup scripts.
2. Predictor.ipynb: Test the deployed endpoint by a serial given sample images.
