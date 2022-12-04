import torch
import cv2 as cv


def box(input, data):
    if len(data) >= 1:
        input = cv.rectangle(input,
                             (int(data['xcenter'][0] - data['width'][0] / 2),
                              int(data['ycenter'][0] + data['height'][0] / 2)),
                             (int(data['xcenter'][0] + data['width'][0] / 2),
                              int(data['ycenter'][0] - data['height'][0] / 2)),
                             (255, 255, 255), 2)
        print('tgt_pos: ' + str(int(data['xcenter'][0])) + ',' + str(int(data['ycenter'][0])))
        return input
    else:
        return input


if __name__ == '__main__':
    model = torch.hub.load('../modules/yolo', 'custom', path='../model/best.pt', source='local')

    cap = cv.VideoCapture(0)
    while True:
        status, img = cap.read()
        if status:
            results = model(img)
            data = results.pandas().xywh[0]
            img = box(img, data)
            out = cv.putText(img, 'press \'q\' to exit \'s\' to save face\n',
                             (0, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
            cv.imshow('1', out)
            if cv.waitKey(1) == ord('s'):
                crops = results.crop(save=True)
            if cv.waitKey(1) == ord('q'):
                break
