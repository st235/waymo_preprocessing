import argparse
import multiprocessing

from functools import partial
from tqdm import tqdm

from file_utils import list_all_files_with_extension
from waymo_utils import process_single_sequence


def process_dataset(files: list,
                    save_dir: str,
                    num_workers: int):
    assert f"num_workers should be positive {num_workers}", \
        num_workers > 0

    scenes = list_all_files_with_extension(files=files, extension='tfrecord')
    scenes_count = len(scenes)

    print(f'Found {scenes_count} scenes.')

    process_scene = partial(
        process_single_sequence,
        save_dir=save_dir
    )

    with multiprocessing.Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(process_scene, scenes), total=scenes_count))


def main():
    parser = argparse.ArgumentParser(
        prog='Waymo Processing Utils',
        description='Unpacks Waymo dataset and packs it in a user-friendly format.')
    parser.add_argument('files', nargs='+')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save modified dataset.')
    parser.add_argument('--num_workers', type=int, default=1,
                        choices=range(1, 32),
                        help='Count of parallel workers.')

    args = parser.parse_args()
    process_dataset(files=args.files, save_dir=args.save_dir, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
