# Generating Multi-View Optical Illusions with Diffusion Models

This notebook generates visual anagrams and other multi-view optical illusions. These are images that change appearance or identity when transformed, such as by a rotation or a permutation of pixels.

### DeepFloyd Access
Our method uses DeepFloyd IF, a pixel-based diffusion model. We do not use Stable Diffusion because latent diffusion models cause artifacts in illusions.

Before using DeepFloyd IF, you must accept its usage conditions. To do so:

* Make sure to have a Hugging Face account and be logged in.
* Accept the license on the model card of DeepFloyd/IF-I-XL-v1.0. Accepting the license on the stage I model card will auto accept for the other IF models.
* Log in locally by entering your Hugging Face Hub access token below.

## Getting Started:
1. Clone the repository:
      ```
      git clone https://github.com/your-username/Generating-Multi-View-Optical-Illusions.git
      cd Generating-Multi-View-Optical-Illusions
      ```
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

3. Set Hugging Face Access Token:
   ```
   export HUGGING_FACE_ACCESS_TOKEN = "your_access_token"
   ```
4. Running the Application:
   ```
   streamlit run app.py
   ```
5. Use the Chatbot:
   * Open your web browser and go to http://localhost:8501 (default Streamlit port).
   * You will see the chat interface titled "Generating Multi-View Optical Illusions Video"

https://github.com/user-attachments/assets/248710f1-cbb0-437d-b67f-b81d36195a52


  

