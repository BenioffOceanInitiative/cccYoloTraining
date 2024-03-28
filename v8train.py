from ultralytics import YOLO
import argparse
import torch

def trainv8(data,epochs):
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        if gpu_count > 0:
            device = list(range(gpu_count))
    else:
        device = 'cpu'
    print(device)
    model = YOLO('yolov8s.pt')
    results = model.train(data=data,epochs=epochs,batch=64,device=device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    trainv8(args.data,args.epochs)