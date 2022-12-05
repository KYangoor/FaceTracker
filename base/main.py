import torch
import cv2 as cv


def box(im, pos):
    if len(pos) >= 1:
        im = cv.rectangle(im,
                          (int(pos['xcenter'][0] - pos['width'][0] / 2),
                              int(pos['ycenter'][0] + pos['height'][0] / 2)),
                          (int(pos['xcenter'][0] + pos['width'][0] / 2),
                              int(pos['ycenter'][0] - pos['height'][0] / 2)),
                          (255, 255, 255), 2)
        print('tgt_pos: ' + str(int(pos['xcenter'][0])) + ',' + str(int(pos['ycenter'][0])))
        return im
    else:
        return im


def from_stream(addr):
    model = torch.hub.load('../modules/yolo', 'custom', path='../model/best.pt', source='local')
    cap = cv.VideoCapture(addr)
    while True:
        status, img = cap.read()
        if status:
            results = model(img)
            data = results.pandas().xywh[0]
            img = box(img, data)
            out = cv.putText(img, 'press \'q\' to exit \'s\' to save face',
                             (0, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
            cv.imshow('1', out)
            if cv.waitKey(1) == ord('s'):
                results.crop(save=True)
            if cv.waitKey(1) == ord('q'):
                break


def from_localimg(addr):
    model = torch.hub.load('../modules/yolo', 'custom', path='../model/best.pt', source='local')
    img = cv.imread(addr)
    results = model(img)
    data = results.pandas().xywh[0]
    img = box(img, data)
    results.show()


if __name__ == '__main__':
    src = input('choose your img source\n'
                '1.camera\n'
                '2.local img\n'
                '3.local video\n\n'
                'input num: ')
    if src == str(1):
        from_stream(0)
    elif src == str(2):
        iaddr = input('input the addr of img: ')
        from_localimg(iaddr)
    elif src == str(3):
        vaddr = input('input the addr of video: ')
        from_stream(0)
    # print(src)
