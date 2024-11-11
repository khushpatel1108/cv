from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from inference_sdk import InferenceHTTPClient
import io
from PIL import Image
import tempfile

# Initialize FastAPI app
app = FastAPI()

# Roboflow client setup
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="nmoxDf7wfxJEIYLkLFDA"
)

# Home route serving upload form
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
        <head>
            <title>PPE Kit Detection</title>
        </head>
        <body>
            <h1>Upload an image for PPE Kit Detection</h1>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
    return html_content

# Endpoint for handling image upload
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_bytes = await file.read()

        # Debug: Check if the file was uploaded successfully
        print("Image uploaded successfully. File size:", len(image_bytes))

        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        # Send the image to Roboflow for inference (passing the file path)
        result = CLIENT.infer(temp_file_path, model_id="ppe-kit-detection-ieadm/3")

        # Check if items are present in the predictions
        items_to_check = ['mask', 'vest', 'shoes', 'gloves', 'helmet', 'googles']
        presence = {item: False for item in items_to_check}

        # Loop through predictions to check for the presence of items
        for pred in result['predictions']:
            if pred['class'] in presence:
                presence[pred['class']] = True

        # Prepare result message
        result_message = "<h2>Detection Results:</h2><ul>"
        for item, is_present in presence.items():
            result_message += f"<li>{item.capitalize()}: {'Present' if is_present else 'Not Present'}</li>"
        result_message += "</ul>"

        return HTMLResponse(content=result_message)

    except Exception as e:
        # Log the error for debugging
        print("Error during upload or inference:", e)
        return HTMLResponse(content=f"<h2>Internal Server Error: {e}</h2>", status_code=500)
