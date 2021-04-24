from tkinter import *
from tkinter import filedialog
import os
import pickle
import matplotlib.pyplot as plt
import cv2

from PIL import ImageTk, Image

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)

tmp = r'app\static\cnnimageretrieval-pytorch\data\test\oxford5k\queries'
tmp = os.path.join(os.getcwd(), tmp)


def open_img():
    global tmp
    x = filedialog.askopenfilename(title='open')
    os.system('python main.py '+x)
    with open('bbx.pkl', 'rb') as f:
        box = pickle.load(f)
    img = Image.open(x)
    img = img.crop(box)
    x = 'q_'+x.split('/')[-1]
    tmp = os.path.join(tmp, x)
    img1 = img.save(tmp)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()

def process():
    global tmp
    k = os.getcwd()
    print(k)
    dst_dir = r'app\static\cnnimageretrieval-pytorch'
    dst_dir = os.path.join(k, dst_dir)
    os.chdir(dst_dir)
    os.system(r'python -m cirtorch.examples.test --network-path retrievalSfM120k-resnet101-gem --datasets oxford5k --whitening retrieval-SfM-120k --query '+tmp)
    fig, axs = plt.subplots(1, 10, figsize=(10 * 3, 10 * 3))
    i = 0
    path = r'D:\WorkSpace\Python\Oxford\app\static\cnnimageretrieval-pytorch\data\test\oxford5k_result.pkl'
    with open(path, 'rb') as f:
        img_paths = pickle.load(f)
    for ax in axs.flat:
        img = cv2.imread(img_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Rank' + str(i), fontdict={'fontsize': 30})
        i += 1
    plt.show()

btn = Button(root, text='open image', command=open_img).pack()
btn = Button(root, text='retrieval', command=process).pack()

root.mainloop()
