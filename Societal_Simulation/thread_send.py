import threading
import queue
import _queue
import time
import traceback
class ThreadSend:
    def __init__(self, thread_num=8):
        self.status = 'init' 
        self.task_list =  queue.Queue()
        self.threads_num = thread_num
        #self.task_num = 0
    def add_task(self, func, args):
        self.task_list.put([func, args])
        #self.task_num += 1
    
    def get_task_num(self):
        return self.task_list.qsize()

    def worker(self):
        #print('start worker', self)
        while True:
            try:
                task = self.task_list.get(timeout=5)
                #print('task',task)
                task[0](*task[1])
                #print('task done',task[0])
                self.task_list.task_done()
                time.sleep(5)
                if self.task_list.empty():
                    break
            except queue.Empty:
                #print(type(e))
                #if not isinstance(e, _queue.Empty):
                #    traceback.print_exc()
                #else:
                #    raise e
                break
        #print('worker end', self)
    
    def start_thread(self):
        #print('start thread ', self.task_list.qsize(),self)
        self.thread_list = []
        for i in range(self.threads_num):
            t = threading.Thread(target=self.worker)
            t.start()
            self.thread_list.append(t)
        #print('wait thread', self)
        self.task_list.join()
        #print('after thread', self)