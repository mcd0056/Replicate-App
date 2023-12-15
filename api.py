from flask import Flask, request, jsonify
import replicate
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if request.method == 'POST':
        data = request.json  # Assuming the request sends data in JSON format

        # Extracting parameters from the request
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        num_outputs = data.get('num_outputs', 1)
        scheduler = data.get('scheduler', 'DDIM')
        num_inference_steps = data.get('num_inference_steps', 50)
        guidance_scale = data.get('guidance_scale', 7.5)
        prompt_strength = data.get('prompt_strength', 0.8)
        refine = data.get('refine', 'expert_ensemble_refiner')
        high_noise_frac = data.get('high_noise_frac', 0.8)
        negative_prompt = data.get('negative_prompt', '')
        prompt = data.get('prompt', '')

        try:
            # Call the replicate API to generate the image
            output = replicate.run(
                "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
                input={
                    "width": width,
                    "height": height,
                    "num_outputs": num_outputs,
                    "scheduler": scheduler,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "prompt_strength": prompt_strength,
                    "refine": refine,
                    "high_noise_frac": high_noise_frac,
                    "negative_prompt": negative_prompt,
                    "prompt": prompt
                }
            )
            if output:
                # Assuming output is a list of image URLs or binary data
                return jsonify({"images": output})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Invalid request"}), 400

if __name__ == '__main__':
    app.run(debug=True)
