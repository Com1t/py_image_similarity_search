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

        # number of image need to process
        self.img_num = self.imgname_queue.qsize()
        # number of processed image
        self.processed_img_num = 0

        for i in range(num_cores):
            self.workers.append(Worker(self.imagelist, self.datapath, self.imgname_queue, self.queue_lock, self.imglist_lock))

        for i in range(len(self.workers)):
            self.workers[i].start()

    def __getitem__(self):
        read_to_idx = self.batch_size
        left_image_num = self.img_num - self.processed_img_num
        if self.img_num == self.processed_img_num:
            return None, None
        elif left_image_num <= self.batch_size:
            # wait until all image within last batch are parsed
            while len(self.imagelist) != left_image_num:
                time.sleep(0.1)
            read_to_idx = left_image_num

        # wait until image list is longer or equal to a batch
        while len(self.imagelist) < read_to_idx:
            time.sleep(0.1)

        image_datalist = self.imagelist[:read_to_idx]
        self.processed_img_num += read_to_idx

        self.imglist_lock.acquire()
        del self.imagelist[:read_to_idx]
        self.imglist_lock.release()

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
    start = time.time()
    dataloader = img_dataloader(r'C:\Users\fuchi\Desktop\working\Buffer', 512)
    num_vec = 0
    image, label = dataloader.__getitem__()
    while image is not None:
        num_vec += len(image)
        print("read ", num_vec)
        image, label = dataloader.__getitem__()

    hash_cal_end = time.time()
    print(f"FPS per sec {num_vec / (hash_cal_end - start)}")
    print("Hash calculation time: ", hash_cal_end - start)