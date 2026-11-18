from fastapi import FastAPI, UploadFile, HTTPException, File
import torch
import uvicorn
from torchvision import transforms
import io
import json
from PIL import Image
import torch.nn as nn

IMG_SIZE = 100

class NaturalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 25 * 25, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


with open("classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)


app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NaturalCNN(len(classes)).to(device)
model.load_state_dict(torch.load("natural_cnn_model.pth", map_location=device))
model.to(device)
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=404, detail="No image")

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            y_pred = torch.argmax(pred).item()

        return {"result": classes[y_pred]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)