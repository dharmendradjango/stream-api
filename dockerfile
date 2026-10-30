FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0
RUN pip install fastapi uvicorn opencv-python-headless onvif-zeep onvif-py3

COPY main.py .
CMD ["python", "main.py"]