import os
from PIL import Image

INPUT_FOLDER = "./pics"
OUTPUT_FOLDER = "./modified_pics"
TARGET_SIZE = (640, 640)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_image(img_path, output_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        original_w, original_h = img.size

        scale = min(TARGET_SIZE[0] / original_w, TARGET_SIZE[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized_img = img.resize((new_w, new_h), Image.LANCZOS)

        background = Image.new("RGB", TARGET_SIZE, (0, 0, 0))

        offset_x = (TARGET_SIZE[0] - new_w) // 2
        offset_y = (TARGET_SIZE[1] - new_h) // 2
        background.paste(resized_img, (offset_x, offset_y))

        background.save(output_path)

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        process_image(input_path, output_path)
        print(f"Immagine {filename} processata")