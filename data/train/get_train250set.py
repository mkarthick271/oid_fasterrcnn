import pandas as pd
import os
import requests
from queue import Queue
from threading import Thread


def download(i, imgcnt, dir, img, url):
    print("Downloading picture {} of {}".format(i+1, imgcnt))
    r = requests.get(url)
    with open(dir+img+".jpg", 'wb') as f:
        f.write(r.content)


class DownloadWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            i, imgcnt, dir, img, url = self.queue.get()
            try:
                download(i, imgcnt, dir, img, url)
            finally:
                self.queue.task_done()

def main():
    data = pd.read_csv("dataset250.csv")
    imgurls = data.loc[:, ['imageid', 'url']]
    imgunq = imgurls.drop_duplicates()
    dir = os.path.abspath(os.path.dirname(__file__)) + '/dataset250/'

    imgtup = [tuple(x) for x in imgunq.values]

    imgcnt = len(imgunq)

    queue = Queue()
    for x in range(8):
        worker = DownloadWorker(queue)
        worker.daemon = True
        worker.start()

    for i in range(imgcnt):
        url = 'https://open-images-dataset.s3.amazonaws.com/train/' + imgtup[i][0] + '.jpg'
        queue.put((i, imgcnt, dir, imgtup[i][0], url))
    queue.join()


if __name__ == '__main__':
    main()

