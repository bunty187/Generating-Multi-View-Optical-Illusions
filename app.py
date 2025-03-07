import streamlit as st
from huggingface_hub import login
import torch
from diffusers import DiffusionPipeline
import mediapy as mp
from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata
from visual_anagrams.animate import animate_two_view
import torchvision.transforms.functional as TF
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import tempfile
import os

# Sidebar for Hugging Face login
with st.sidebar:
    st.header("Hugging Face Login")
    token = st.text_input("Enter your Hugging Face token", type="password")
    if token:
        login(token=token)
        st.success("Logged in successfully!")

# Streamlit app layout
st.title("Prompt-Based Video Generation")

# Prompt input
prompt_1 = st.text_input("Enter the first prompt", "painting of a snowy mountain village")
prompt_2 = st.text_input("Enter the second prompt", "painting of a horse")

if st.button("Generate Video"):
    if not token:
        st.error("Please log in to Hugging Face using the token in the sidebar.")
    else:
        # Define device
        device = 'cuda'

        def im_to_np(im):
            im = (im / 2 + 0.5).clamp(0, 1)
            im = im.detach().cpu().permute(1, 2, 0).numpy()
            im = (im * 255).round().astype("uint8")
            return im

        # Load pipelines
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16).to(device)
        stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16).to(device)
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16).to(device)

        # Views
        views = get_views(['identity', 'rotate_cw'])

        # Embed prompts
        prompts = [prompt_1, prompt_2]
        prompt_embeds = [stage_1.encode_prompt(prompt) for prompt in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)

        # Stage 1
        image_64 = sample_stage_1(stage_1, prompt_embeds, negative_prompt_embeds, views, num_inference_steps=30, guidance_scale=10.0, reduction='mean', generator=None)
        image_256 = sample_stage_2(stage_2, image_64, prompt_embeds, negative_prompt_embeds, views, num_inference_steps=30, guidance_scale=10.0, reduction='mean', noise_level=50, generator=None)
        image_1024 = stage_3(prompt=prompts[0], image=image_256, noise_level=0, output_type='pt', generator=None).images
        image_1024 = image_1024 * 2 - 1

        # Save video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            save_video_path = temp_file.name

        pil_image = TF.to_pil_image(image_1024[0] / 2. + 0.5)
        animate_two_view(pil_image, views[1], prompt_1, prompt_2, save_video_path=save_video_path, hold_duration=120, text_fade_duration=10, transition_duration=45, im_size=image_1024.shape[-1], frame_size=int(image_1024.shape[-1] * 1.5))

        # Convert video to MP4
        try:
            clip = VideoFileClip(save_video_path)
            converted_path = save_video_path.replace(".mp4", "_converted.mp4")
            clip.write_videofile(converted_path, codec='libx264')
            st.video(converted_path)
        except Exception as e:
            st.error(f"An error occurred while converting the video: {e}")
        
        # Clean up
        os.remove(save_video_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)
