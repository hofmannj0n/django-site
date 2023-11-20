import streamlit as st
import boto3
import time

# Replace 'your_access_key', 'your_secret_key', and 'your_region' with your AWS credentials and region
aws_access_key_id = 'AKIARXUYLPUEM72JFS34'
aws_secret_access_key = 'UBjXgn1VUu/C6xxWVqRMro9D10Rn2MToAPK8qboj'
region_name = 'us-east-1'


# Replace 'your_s3_bucket_name' with your S3 bucket name
s3_bucket_name = 'bucket-feather'

# Initialize S3 and Lambda clients
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=region_name)

lambda_client = boto3.client('lambda', aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key,
                             region_name=region_name)

def upload_to_s3(file):
    # Upload the file to your S3 bucket
    s3_object_key = f"uploads/]{file.name}"  # Modify as needed
    s3_client.upload_fileobj(file, s3_bucket_name, s3_object_key)
    return s3_object_key

def invoke_lambda_function(s3_object_key):
    # Invoke your Lambda function
    response = lambda_client.invoke(
        FunctionName='feather-lambda',
        InvocationType='Event',  # Asynchronous invocation
        Payload=f'{{"s3_object_key": "{s3_object_key}"}}'
    )

    # Return the Lambda invocation response
    return response

# Streamlit UI
st.title('PDF Upload and Processing')

uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF selected for processing.")

    if st.button('Upload and Process PDF'):
        # Upload the file to S3
        s3_object_key = upload_to_s3(uploaded_file)
        st.success(f"PDF uploaded successfully to S3 with key: {s3_object_key}")

        # Invoke the Lambda function asynchronously
        response = invoke_lambda_function(s3_object_key)
        st.success("Lambda function invoked asynchronously. Check Lambda logs for processing status.")

        # Provide a download link for the Lambda function output (modify as needed)
        output_link = f"https://s3.{region_name}.amazonaws.com/{s3_bucket_name}/{s3_object_key}-output.txt"
        st.subheader("Lambda Function Output:")
        st.markdown(f"Download the output file: [{output_link}]({output_link})")
