import replicate
import streamlit as st
import requests
import zipfile
import io
import os

# API Keys - .streamlit/secrets.toml
REPLICATE_MODEL_ENDPOINT = st.secrets["REPLICATE_MODEL_ENDPOINT"]
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]


def generate_image_page():

    st.title("Generate Images from Stable Diffusion 😍")
    
    st.markdown("Create High Quailty Images from Text")

    # Session State for Image Generation
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None


    # Main Form
    with st.form("my_form"):
        st.info("Start by entering a prompt")
        with st.expander("Refine your output here"):

            # Advanced Settings to adjust settings for the image generation
            width = st.number_input("Width of output image", value=1024)
            height = st.number_input("Height of output image", value=1024)
            num_outputs = st.slider(
                "Number of images to output", value=1, min_value=1, max_value=4)
            scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
            num_inference_steps = st.slider(
                "Number of denoising steps", value=50, min_value=1, max_value=500)
            guidance_scale = st.slider(
                "Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
            prompt_strength = st.slider(
                "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of information in image)", value=0.8, max_value=1.0, step=0.1)
            refine = st.selectbox(
                "Select refine style to use (left out the other 2)", ("expert_ensemble_refiner", "None"))
            high_noise_frac = st.slider(
                "Fraction of noise to use for `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
        
        prompt = st.text_area(
            "Enter prompt: start typing",
            value="Add your best prompt here!"
        )
        negative_prompt = st.text_area(
            "Enter Anything you dont want in the picture, Examples Below",
            value="the absolute worst quality, distorted features",
            help="This is a negative prompt, basically type what you don't want to see in the generated image"
        )
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)



    # Image Generation
    if submitted:
        with st.spinner('👩🏾‍🍳 Whipping up your words into art...'):
            try:
                # Calling the replicate API to get the image
                all_images = []  # List to store all generated images
                output = output = replicate.run(REPLICATE_MODEL_ENDPOINT,
  input={
    "width": width,
    "height": height,
    "prompt": prompt,
    "scheduler": scheduler,
    "num_outputs": num_outputs,
    "guidance_scale": guidance_scale,
    "num_inference_steps": num_inference_steps,
    "refine": refine,
    "high_noise_frac": high_noise_frac,
    "negative_prompt": negative_prompt,
    "prompt_strength": prompt_strength
  }
)
                if output:
                    st.toast('Your image has been generated!', icon='😍')
                    # Displaying the image
                    for image in output:
                        st.image(image, caption="Generated Image 🎈", use_column_width=True)
                        # Add image to the list
                        all_images.append(image)

                    # Create a BytesIO object
                    zip_io = io.BytesIO()

                    # Download option for each image
                    with zipfile.ZipFile(zip_io, 'w') as zipf:
                        for i, image in enumerate(all_images):
                            response = requests.get(image)
                            if response.status_code == 200:
                                image_data = response.content
                                # Write each image to the zip file with a name
                                zipf.writestr(f"output_file_{i+1}.png", image_data)
                            else:
                                st.error(
                                    f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}"
                                )

                    # Create a download button for the zip file
                    st.download_button(
                        "Download All Images", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True
                    )
            except Exception as e:
                st.error(f'Encountered an error: {e}')

if __name__ == "__main__":
    generate_image_page()                   
