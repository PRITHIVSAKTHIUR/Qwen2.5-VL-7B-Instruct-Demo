import gradio as gr
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer
from transformers.image_utils import load_image
from threading import Thread
import time
import torch
import spaces

# -----------------------
# Progress Bar Helper
# -----------------------
def progress_bar_html(label: str) -> str:
    """
    Returns an HTML snippet for a thin progress bar with a label.
    The progress bar is styled as a dark red animated bar.
    """
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #9370DB; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #4B0082; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" #else ; MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda").eval()

@spaces.GPU
def model_inference(input_dict, history):
    text = input_dict["text"]
    files = input_dict["files"]

    # Load images if provided
    if len(files) > 1:
        images = [load_image(image) for image in files]
    elif len(files) == 1:
        images = [load_image(files[0])]
    else:
        images = []

    # Validate input
    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")
        return
    if text == "" and images:
        gr.Error("Please input a text query along with the image(s).")
        return

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text},
            ],
        }
    ]

    # Apply chat template and process inputs
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Set up streamer for real-time output
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output
    buffer = ""
    yield progress_bar_html("Processing with Qwen2.5VL Model")
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer


# Example inputs
examples = [
    [{"text": "Describe the document?", "files": ["example_images/document.jpg"]}],
    [{"text": "What does this say?", "files": ["example_images/math.jpg"]}],
    [{"text": "What is this UI about?", "files": ["example_images/s2w_example.png"]}],
    [{"text": "Where do the severe droughts happen according to this diagram?", "files": ["example_images/examples_weather_events.png"]}],

]

demo = gr.ChatInterface(
    fn=model_inference,
    description="# **Qwen2.5-VL-7B-Instruct**",
    examples=examples,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple"),
    stop_btn="Stop Generation",
    multimodal=True,
    cache_examples=False,
)

demo.launch(debug=True)