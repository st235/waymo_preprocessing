import argparse

from file_utils import list_all_files_with_extension
from waymo_utils import process_single_sequence


def process_dataset(files: list,
                    save_path: str):
    scenes = list_all_files_with_extension(files=files, extension='tfrecord')

    for scene in scenes:
        print(f"Processing scene {scene}")

        process_single_sequence(sequence_file=scene,
                                save_path=save_path)


def main():
    parser = argparse.ArgumentParser(
        prog='Waymo Processing Utils',
        description='Unpacks Waymo dataset and packs it in a user-friendly format.')
    parser.add_argument('files', nargs='+')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save modified dataset.')

    args = parser.parse_args()
    process_dataset(files=args.files, save_path=args.save_dir)


if __name__ == '__main__':
    main()
