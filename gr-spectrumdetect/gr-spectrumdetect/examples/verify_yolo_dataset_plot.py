import cv2
import glob

path_dir_img = './datasets/impaired/images/val/'
path_dir_label = './datasets/impaired/labels/val/'
image_files = glob.glob(path_dir_img+'*.png')
num_cnt = 0
for img_file in image_files[:1]:
    print(img_file)
    print('press 0 to close')
    img = cv2.imread(img_file)
    dh, dw, _ = img.shape

    fl = open(path_dir_label+(img_file.split('.png')[-2]+'.txt').split('/')[-1])
    data = fl.readlines()
    fl.close()

    for dt in data:

        _, x, y, w, h = map(float, dt.split(' '))

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
    
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)

    cv2.imshow(img_file,img)
    cv2.waitKey(0)
    #cv2.imwrite(str(num_cnt)+'.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.destroyAllWindows()
    num_cnt += 1
