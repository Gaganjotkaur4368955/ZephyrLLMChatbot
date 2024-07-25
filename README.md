``# ZephyrLLMChatbot: Focus on your goals

A guideline to build a no-cost LLM chatbot focused on promoting real estate knowlegde and positive interactions with the user.

Introduction
This project demonstrates how to create a customized LLM chatbot using Hugging Face's Inference API and Gradio. The chatbot, named "Focus on your goals❤️🕊️", is designed to help the users to focus on their goals and manage stress and help you with your studies.

Prerequisites
Before you start, make sure you have the following:

Python 3.6 or higher
A Hugging Face account (sign up at huggingface.co)
Installation
Clone this repository:

git clone https://github.com/Gaganjotkaur4368955/ZephyrLLMChatbot.git
cd ZephyrLLMChatbot
Install the required packages:

pip install -r requirements.txt
Usage
To run the chatbot locally:

Open a terminal and navigate to the project directory.
Run the following command:
python app.py
Open your web browser and go to the URL displayed in the terminal (usually http://127.0.0.1:7860).
Features
Uses the Zephyr-7b-beta model from Hugging Face for natural language processing.
Customizable system message to set the chatbot's behavior.
Adjustable parameters:
Max new tokens
Temperature
Top-p (nucleus sampling)
Example prompts to demonstrate the chatbot's capabilities.
Customization
You can customize the chatbot by modifying the following in app.py:

System message: Edit the system_message variable to change the chatbot's overall behavior.
Model: Change the InferenceClient to use a different model from Hugging Face.
Examples: Add or modify the examples list to showcase different interactions.
Interface: Adjust the Gradio interface settings for a different look and feel.
Deployment
This project can be easily deployed on Hugging Face Spaces:

Fork this repository to your GitHub account.
Go to huggingface.co/spaces and click "Create new Space".
Choose "Gradio" as the SDK.
Link your forked GitHub repository.
Deploy and enjoy your chatbot!
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Contact
For any questions or feedback, please reach out to ga4368965@alphacollege.me. `
