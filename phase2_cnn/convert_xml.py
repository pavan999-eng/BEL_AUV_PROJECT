import os
import xml.etree.ElementTree as ET

# Class mapping
classes = {
    "ball": 0,
    "plane": 1,
    "cylinder": 2,
    "cube": 3,
    "tyre": 4,
    "circle cage": 5,
    "human body": 6,
    "metal bucket": 7,
    "rov": 8,
    "square cage": 9
}


def convert_xml(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for xml_file in os.listdir(input_folder):

        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(input_folder, xml_file))
        root = tree.getroot()

        size = root.find("size")

        width = int(size.find("width").text)
        height = int(size.find("height").text)

        yolo_data = []

        for obj in root.findall("object"):

            class_name = obj.find("name").text.lower()

            if class_name not in classes:
                continue

            class_id = classes[class_name]

            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # YOLO format conversion
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_data.append(
                f"{class_id} {x_center} {y_center} {box_width} {box_height}"
            )

        txt_filename = xml_file.replace(".xml", ".txt")

        with open(os.path.join(output_folder, txt_filename), "w") as f:
            f.write("\n".join(yolo_data))


# Convert train labels
convert_xml(
    "dataset/raw_labels_train",
    "dataset/labels/train"
)

# Convert validation labels
convert_xml(
    "dataset/raw_labels_val",
    "dataset/labels/val"
)

print("Train and validation XML converted successfully.")