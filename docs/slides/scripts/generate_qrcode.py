#!/usr/bin/env python3

import qrcode
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path

def generate_qr_codes():
    # Configuration
    links = {
        "Study Notes": "https://567-labs.github.io/systematically-improving-rag/",
        "Talks": "https://567-labs.github.io/systematically-improving-rag/talks/",
        "Slack": "https://join.slack.com/t/improvingrag/shared_invite/zt-3dkinqb3q-vknvaBLoTx5tBj4PpGOVjw"
    }

    # QR code settings
    qr_size = 400
    padding = 20
    text_height = 60

    # Create individual QR codes
    qr_images = []

    for label, url in links.items():
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Create QR image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img = qr_img.resize((qr_size, qr_size), Image.Resampling.LANCZOS)

        # Create image with label
        total_height = qr_size + text_height + padding
        img = Image.new('RGB', (qr_size + 2 * padding, total_height), 'white')

        # Paste QR code
        img.paste(qr_img, (padding, padding))

        # Add label
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a nice font, fallback to default if not available
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except:
            font = ImageFont.load_default()

        # Center the text
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (img.width - text_width) // 2
        text_y = qr_size + padding + 10

        draw.text((text_x, text_y), label, fill='black', font=font)

        qr_images.append(img)

    # Combine all QR codes into one image
    total_width = sum(img.width for img in qr_images) + padding * 2
    max_height = max(img.height for img in qr_images)

    combined = Image.new('RGB', (total_width, max_height), 'white')

    x_offset = padding
    for img in qr_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    output_path = Path(__file__).parent.parent / "assets" / "images" / "codes.jpeg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, 'JPEG', quality=95)
    print(f"QR codes saved to: {output_path}")

    # Save configuration
    config = {
        "links": links,
        "qr_code_settings": {
            "size": qr_size,
            "padding": padding,
            "text_height": text_height,
            "error_correction": "HIGH"
        },
        "output": str(output_path)
    }

    config_path = output_path.parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    generate_qr_codes()