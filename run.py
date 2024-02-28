# Created by Baole Fang at 2/27/24

from multiprocessing import Process, Queue
import os
import argparse


def worker(q: Queue, root: str, output: str, gpu: str):
    while not q.empty():
        job = q.get()
        model = os.path.join(output, job)
        print(f'Processing {model} on GPU {gpu}')
        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python train.py -s {root} -i {job} -m {model} >/dev/null 2>&1')
        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration 7000 -m {model} >/dev/null 2>&1')
        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration 30000 -m {model} >/dev/null 2>&1')
        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {model} >/dev/null 2>&1')


def main(root: str, output: str, gpus: str):
    q = Queue()
    jobs = set()
    for name in os.listdir(os.path.join(root, 'img')):
        jobs.add(name.rsplit('_', maxsplit=1)[0])
    for job in jobs:
        q.put(job)
    processes = []
    for gpu in gpus.split(','):
        p = Process(target=worker, args=(q, root, output, gpu))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    main(args.input, args.output, args.gpus)
