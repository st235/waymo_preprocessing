import argparse
import concurrent.futures

from file_utils import list_all_files_with_extension
from waymo_utils import process_single_sequence


def process_batch(batch_of_scenes: list[str],
                  save_dir: str,
                  worker_id: int):
    print(f"Worker #{worker_id:03d} started with a batch: {batch_of_scenes} of size {len(batch_of_scenes)}")

    for i, scene in enumerate(batch_of_scenes):
        progress = i / len(batch_of_scenes) * 100.0
        print(f"Processing scene {scene}, worker #{worker_id:03d} progress {progress:.2f}%")

        process_single_sequence(sequence_file=scene,
                                save_dir=save_dir)

    print(f"Worker #{worker_id:03d} finished.")


def process_dataset(files: list,
                    save_dir: str,
                    num_workers: int):
    assert f"num_workers should be positive {num_workers}", \
        num_workers > 0

    scenes = list_all_files_with_extension(files=files, extension='tfrecord')
    scenes_count = len(scenes)

    print(f'Found {scenes_count} scenes.')

    batches = list()
    batch_size = scenes_count // num_workers

    for batch_index in range(num_workers):
        batch_start = batch_index * batch_size
        batch_finish = (batch_index + 1) * batch_size

        if batch_index + 1 == num_workers and scenes_count % num_workers > 0:
            # Last batch will be a bit bigger if there is not enough elements
            # at the end to end up in a standalone batch.
            batch_finish = len(scenes)

        batches.append(scenes[batch_start:batch_finish])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_batch, batches, [save_dir] * num_workers, range(1, num_workers + 1))


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
