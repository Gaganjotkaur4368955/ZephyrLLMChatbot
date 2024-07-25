import gradio as gr
from huggingface_hub import InferenceClient
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are a good listener. You advise relaxation exercises, suggest avoiding negative thoughts, and guide through steps to manage stress. Discuss what's on your mind, or ask me for a quick relaxation exercise."
    messages = [{"role": "system", "content": system_message}]

    for val in history:

    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are a good presenter. You advise study material, suggest study schedule, and guide through steps to manage stress. Discuss what's your goal, or ask me about my track of study.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["I feel positive about my studies."],
        ["Can you guide me about my carrer options?"],
        ["How do I ignore the negative facts and remain positive?"]
    ],
    title = 'Focus on your goals❤️🕊️'
)
if __name__ == "__main__":
    demo.launch()
