# generate_mock_bol.py
from PIL import Image, ImageDraw
import os

def create_scanned_bol(output_path="mock_scanned_bol.pdf"):
    # Create a blank white canvas simulating an A4 page
    width, height = 800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    y_offset = 50
    
    def write_line(text, size_multiplier=1, indent=0):
        nonlocal y_offset
        draw.text((50 + indent, y_offset), text, fill="black")
        y_offset += int(25 * size_multiplier)

    # Document Header
    write_line("BILL OF LADING (BOL)", size_multiplier=2)
    write_line("=" * 60)
    write_line("BOL NUMBER: BOL-99281-X", size_multiplier=1.2)
    write_line("DATE: 2026-03-31")
    write_line("-" * 60)
    
    # Shipper & Consignee Details
    write_line("SHIPPER / EXPORTER:", size_multiplier=1.1)
    write_line("  Global Freight Solutions Inc.", indent=20)
    write_line("  100 Industrial Parkway, Sector 4", indent=20)
    write_line("  Port of Newark, NJ 07114, USA", indent=20)
    write_line("-" * 60)
    
    write_line("CONSIGNEE (RECEIVER):", size_multiplier=1.1)
    write_line("  EuroDistribution GmbH", indent=20)
    write_line("  Industriestrasse 42, Block B", indent=20)
    write_line("  Hamburg 20457, Germany", indent=20)
    write_line("=" * 60)
    
    # Shipment Details / Table
    write_line("LINE ITEMS:", size_multiplier=1.2)
    write_line("QTY | DESCRIPTION                           | WEIGHT (KG) | VOLUME (CBM)")
    write_line("-" * 60)
    write_line("18  | Industrial Lithium-Ion Battery Packs   | 4,200.00    | 12.5")
    write_line("04  | Electrical Control Cabinets           | 1,850.00    | 6.2")
    write_line("22  | Steel Mounting Brackets (Boxes)       | 450.00      | 1.8")
    write_line("-" * 60)
    write_line("TOTAL WEIGHT: 6,500.00 KG")
    write_line("TOTAL VOLUME: 20.50 CBM")
    write_line("=" * 60)
    
    # Footer
    write_line("PORT OF LOADING: Port of New York/New Jersey (USNYC)")
    write_line("PORT OF DISCHARGE: Port of Hamburg (DEHAM)")
    write_line("CARRIER: Atlantic Ocean Lines")
    write_line("VESSEL: Ocean Titan | VOYAGE: 412E")
    
    # Save image as PDF
    image.save(output_path, "PDF", resolution=100.0)
    print(f"Successfully generated visual-only scanned PDF at: {output_path}")

if __name__ == "__main__":
    create_scanned_bol()