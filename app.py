import streamlit as st
import torch
from diffusers import DiffusionPipeline
from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import im_to_np
import mediapy as mp
from visual_anagrams.animate import animate_two_view
import torchvision.transforms.functional as TF
from moviepy.editor import VideoFileClip
import os

# Streamlit app
def main():
    st.title("Visual Anagrams Video Generator")

    # Input prompts
    prompt_1 = st.text_input("Enter the first prompt", "painting of a snowy mountain village")
    prompt_2 = st.text_input("Enter the second prompt", "painting of a horse")

    if st.button("Generate Video"):
        if not prompt_1 or not prompt_2:
            st.error("Please provide both prompts.")
            return

        # Load models
        device = 'cuda'
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

        # Generate images
        image_64 = sample_stage_1(stage_1, prompt_embeds, negative_prompt_embeds, views, num_inference_steps=30, guidance_scale=10.0)
        image_256 = sample_stage_2(stage_2, image_64, prompt_embeds, negative_prompt_embeds, views, num_inference_steps=30, guidance_scale=10.0, noise_level=50)
        image_1024 = stage_3(prompt=prompt_1, image=image_256, noise_level=0, output_type='pt').images
        image_1024 = image_1024 * 2 - 1

        # Save video
        save_video_path = './animation1.mp4'
        pil_image = TF.to_pil_image(image_1024[0] / 2. + 0.5)
        animate_two_view(pil_image, views[1], prompt_1, prompt_2, save_video_path=save_video_path, hold_duration=120, text_fade_duration=10, transition_duration=45, im_size=image_1024.shape[-1], frame_size=int(image_1024.shape[-1] * 1.5))

        # Convert to MP4
        try:
            clip = VideoFileClip(save_video_path)
            clip.write_videofile('./animation_converted.mp4', codec='libx264')
            save_video_path = './animation_converted.mp4'
        except Exception as e:
            st.error(f"An error occurred while converting the video: {e}")
            return

        # Display video
        st.video(save_video_path)

if __name__ == "__main__":
    main()
