import torch
import numpy as np
import cv2
import os
import shutil
import streamlit as st
from tempfile import NamedTemporaryFile
from detect import run
from models.common import DetectMultiBackend
from models.classification import Classification_model
from utils.torch_utils import select_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lấy đường dẫn đến file hiện tại
current_file_path = os.path.abspath(__file__)
ROOT = os.path.dirname(current_file_path)
print(ROOT)

class Params:
    def __init__(self):
        self.detection_weights = os.path.join(ROOT, 'detection_weights.pt')
        self.classification_weights = os.path.join(ROOT, 'classification_weights.pt')
        self.processing_dir = os.path.join(ROOT, 'Demo')
        self.detection_name = 'exp'
        self.boxes_dir = os.path.join(self.processing_dir, self.detection_name, 'labels', 'frame.txt')
        self.data = os.path.join(ROOT, 'data', 'coco.yaml')


        self.detection_model = DetectMultiBackend(self.detection_weights, device=device, dnn=False, data=self.data, fp16=False)
        self.classification_model = Classification_model(self.classification_weights, num_classes=7, device=device)

params = Params()

def delete_file(folder_path):
    # Kiểm tra nếu đường dẫn là một tệp
    if os.path.isfile(folder_path):
        os.remove(folder_path)

    # Kiểm tra nếu đường dẫn là một thư mục
    elif os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            delete_file(file_path)
        os.rmdir(folder_path)

def detection(source, frame):
      #command = f'python {params.detection} --weights {params.weights} --source {source} --project {params.processing_dir} --name {params.detection_name} --save-txt --nosave'
      run(model = params.detection_model, weights= params.detection_weights, source=source, project=params.processing_dir, name = params.detection_name, save_txt = True, nosave= True)
      boxes_dir = params.boxes_dir

      # đọc file boundingbox vừa được tạo ra
      with open(boxes_dir, 'r') as file:
          lines = file.readlines()

      # Tách từng dòng và xử lý từng hàng
      processed_lines = []
      for line in lines:
          # Tách các giá trị trong dòng, bỏ qua các khoảng trắng không cần thiết
          values = line.strip().split()
          # Chuyển các giá trị từ chuỗi sang kiểu float (nếu cần)
          float_values = [float(value) for value in values]
          x, y, w, h = float_values[1:5]
          x, y, w, h = map(int, [x * frame.shape[1], y * frame.shape[0],
                      w * frame.shape[1], h * frame.shape[0]])
          processed_lines.append([x, y, w, h])
      delete_file(params.processing_dir + '/' + params.detection_name)
      return processed_lines

def draw_bounding_boxes(frame, bounding_boxes, labels):
    # Vẽ bounding boxes trên frame
    for bbox,label in zip(bounding_boxes, labels):
        x, y, w, h = bbox
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def cut_image(frame, bounding_boxes, gain = 1.02, pad = 10):
    face_images = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x1 + w, y1 + h
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (100, 100))
        face_images.append(crop)
    return face_images

def classification(model, face_images):
    class_labels = np.array(['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'])
    x_tensor = torch.tensor(face_images, dtype=torch.float32).permute(0, 3, 1, 2).to(select_device(''))
    output = model(x_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

    label_sto = []
    for i in range(len(predicted_classes)):
        label_sto.append(class_labels[predicted_classes[i].item()])
    return label_sto

def write_to_annotation_file(file_path, bounding_boxes, labels, frame_idx):
    with open(file_path, 'a') as file:
        for bbox, label in zip(bounding_boxes, labels):
            x, y, w, h = bbox
            file.write(f"{frame_idx} {x} {y} {w} {h} {label}\n")

def create_video_output(video, video_output_dir):
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo đối tượng VideoWriter để tạo video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Xác định codec
    return cv2.VideoWriter(video_output_dir, fourcc, fps, (width, height))

def main(video_dir, output_dir):

    file_output_dir = os.path.join(output_dir, 'annotation.txt')
    video_output_dir = os.path.join(output_dir, 'video_output.mp4')
    # Đọc và lấy số frame của video hiện tại
    video = cv2.VideoCapture(video_dir)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    shutil.rmtree(os.path.join(ROOT, 'Demo'))
    os.makedirs(os.path.join(ROOT, 'Demo'))
    out = create_video_output(video, video_output_dir)


    for frame_idx in range(num_frames):
        if (frame_idx % 10 == 0):
            print(f"Processing frame {frame_idx} of {num_frames}")
        # Đọc frame
        ret, frame = video.read()
        if not ret:
            break
        frame_path = os.path.join(params.processing_dir, 'frame' + '.jpg')
        cv2.imwrite(frame_path, frame)

        # Chạy detection
        bounding_boxes = detection(frame_path, frame)

        #Cắt cận ảnh khuôn mặt và resize
        face_images = cut_image(frame, bounding_boxes)

        #Phân loại cảm xúc
        labels = classification(params.classification_model, face_images)

        #Ghi lại thông tin vào file annotation.txt
        write_to_annotation_file(file_output_dir, bounding_boxes, labels, frame_idx)

        #Vex bounding boxes và labels cho frame hiện tại
        drawn_frame = draw_bounding_boxes(frame, bounding_boxes, labels)

        #Thêm frame đã được vẽ bounding boxes và labels vào video output
        out.write(frame)

        # Frame sau khi xử lí xong
        os.remove(frame_path)

    # Đóng video
    video.release()
    out.release()

    print(f"Bounding boxes and labels saved to {file_output_dir}")
    print(f"Results video saved to {video_output_dir}")
    return file_output_dir, video_output_dir


# if __name__ == '__main__': 
#     video_dir = r'D:\UIT\HocTap\Nam2\HK2\CS117\yolov9-face-detection-Copy\yolov9\WIN_20240610_18_31_30_Pro.mp4'
#     main(video_dir)

if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
if 'annotations_bytes' not in st.session_state:
    st.session_state.annotations_bytes = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'annotations_path' not in st.session_state:
    st.session_state.annotations_path = None
    
st.title("Facial Expression Capture")

path_output = st.text_input("Enter video path:", value=os.path.join(ROOT, 'Demo'))

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    temp_video_file = NamedTemporaryFile(delete=False)
    temp_video_file.write(uploaded_file.read())
    temp_video_file.close()

    st.video(temp_video_file.name)  

    if st.button("Get Output"):
        st.session_state.annotations_path, st.session_state.video_path = main(temp_video_file.name, path_output)

        if os.path.exists(st.session_state.video_path):
            with open(st.session_state.video_path, 'rb') as video_file:
                st.session_state.video_bytes = video_file.read()
        else:
            st.error(f"Video file {st.session_state.video_path} not found.")

        if os.path.exists(st.session_state.annotations_path):
            with open(st.session_state.annotations_path, "rb") as annotations_file:
                st.session_state.annotations_bytes = annotations_file.read()
        else:
            st.error(f"Annotations file {st.session_state.annotations_path} not found.")

if st.session_state.video_bytes:
    st.download_button(
        label="Download Video",
        data=st.session_state.video_bytes,
        file_name="emotion_video.mp4",
        mime="video/mp4"
    )

if st.session_state.annotations_bytes:
    annotation_name = 'Annotation'
    st.download_button(
        label=f"Download {annotation_name}",
        data=st.session_state.annotations_bytes,
        file_name=f'{annotation_name}.txt',
        mime="text/plain"
    )