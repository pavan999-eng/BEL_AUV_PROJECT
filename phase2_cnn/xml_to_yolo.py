import os
import xml.etree.ElementTree as ET

# Paths
xml_dir = "/Users/pavankumarvaranasi/Downloads/21331143/UATD_Test_2/annotations"
output_dir = "/Users/pavankumarvaranasi/Downloads/21331143/UATD_Test_2/labels"

os.makedirs(output_dir, exist_ok=True)

# Your classes (must match training order)
classes = [
    "ball",
    "plane",
    "cylinder",
    "cube",
    "tyre",
    "circle cage",
    "human body",
    "metal bucket",
    "rov",
    "square cage"
]

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    x *= dw
    w *= dw
    y *= dh
    h *= dh

    return (x, y, w, h)

for xml_file in os.listdir(xml_dir):

    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_file = os.path.join(
        output_dir,
        xml_file.replace(".xml", ".txt")
    )

    with open(txt_file, "w") as f:

        for obj in root.iter("object"):

            cls = obj.find("name").text

            if cls not in classes:
                continue

            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")

            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text)
            )

            bb = convert((w, h), b)

            f.write(
                str(cls_id) + " " +
                " ".join([str(a) for a in bb]) + "\n"
            )

print("Conversion completed!")