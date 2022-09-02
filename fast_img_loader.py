import torch
import os
from PIL import Image
from torchvision import transforms
import time
import threading
import queue

torch.manual_seed(1)  # reproducible
num_cores = 12


class img_dataloader():
    def __init__(self, datapath, batch_size):
        # config & image read
        self.imagelist = []
        self.datapath = datapath
        self.batch_size = batch_size

        # multithread
        self.workers = []
        self.queue_lock = threading.Lock()
        self.imglist_lock = threading.Lock()
        self.imgname_queue = queue.Queue()

        self.img_extensions = ['jpeg', 'psd', 'jpg', 'png', 'gif']
        for file in os.listdir(self.datapath):
            filename = os.fsdecode(file)
            if filename.split('.')[-1].lower() in self.img_extensions:
                self.imgname_queue.put(filename)

        self.img_num = self.imgname_queue.qsize()

        for i in range(num_cores):
            self.workers.append(Worker(self.imagelist, self.datapath, self.imgname_queue, self.queue_lock, self.imglist_lock))

        for i in range(len(self.workers)):
            self.workers[i].start()

    def __getitem__(self):
        read_to_idx = self.batch_size
        if self.imgname_queue.qsize() == 0:
            return None, None
        elif self.imgname_queue.qsize() <= self.batch_size:
            read_to_idx = self.imgname_queue.qsize()

        while len(self.imagelist) < read_to_idx:
            time.sleep(0.1)
            print("in loop ", len(self.imagelist), "IDX: ", read_to_idx)

        image_datalist = self.imagelist[:read_to_idx]

        self.imglist_lock.acquire()
        print("acquired ", len(self.imagelist))
        del self.imagelist[:read_to_idx]
        self.imglist_lock.release()
        print("release ", len(self.imagelist))

        return [x[0] for x in image_datalist], [y[0] for y in image_datalist]

    def __len__(self):
        return self.img_num

    def stop_dataloader(self):
        for i in range(len(self.workers)):
            self.workers[i].join()


class Worker(threading.Thread):
    def __init__(self, imagelist, datapath, imgname_queue, queue_lock, imglist_lock):
        threading.Thread.__init__(self)
        self.imagelist = imagelist
        self.datapath = datapath
        self.imgname_queue = imgname_queue
        self.queue_lock = queue_lock
        self.imglist_lock = imglist_lock

        # transform
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def run(self):
        while self.imgname_queue.qsize() > 0:
            self.queue_lock.acquire()
            label = self.imgname_queue.get()
            self.queue_lock.release()

            image = Image.open(os.path.join(self.datapath, label)).convert('RGB')
            image = self.transforms(image)

            self.imglist_lock.acquire()
            self.imagelist.append((image, label))
            self.imglist_lock.release()


if __name__ == "__main__":
    dataloader = img_dataloader(r'C:\Users\fuchi\Desktop\working\abc', 256)
    image, label = dataloader.__getitem__()
    while image is not None:
        print("read ", len(image))
        image, label = dataloader.__getitem__()

