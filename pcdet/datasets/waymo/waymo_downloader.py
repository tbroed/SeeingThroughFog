import os
import glob
import time
import logging
import argparse
import datetime

from tqdm import tqdm
from pathlib import Path

DOWNLOAD_FOLDER = '/srv/beegfs02/scratch/tracezuerich/data/datasets/WaymoOpenDataset/waymo_open_dataset_v_1_2_0'

gsutil = '/home/mhahner/scratch/apps/anaconda3/envs/PCDet/bin/gsutil'

def extract_all_tar_files(source_folder: str, additional_sub_folder: str = None) -> None:

    tar_files = []
    domain_adapation_tar_files = []

    domain_adapation_available = 'domain_adaptation' in os.listdir(source_folder)

    all_tar_files = sorted(glob.glob(source_folder + '/**/*.tar', recursive=True))

    if domain_adapation_available:

        for tar_file in all_tar_files:

            if 'domain_adaptation' in tar_file:

                domain_adapation_tar_files.append(tar_file)

            else:

                tar_files.append(tar_file)

    else:

        tar_files = all_tar_files

    for tar_list in [tar_files, domain_adapation_tar_files]:

        p_bar = tqdm(tar_list)

        for tar_file in p_bar:

            p_bar.set_description(tar_file)

            current_folder = Path(tar_file).parent
            current_file = Path(tar_file).name

            os.system(f'cd {current_folder}; tar xf {current_file}')

            if additional_sub_folder:

                if 'unlabeled' in tar_file:

                    parent_folder = Path(tar_file).parent.parent.parent

                else:

                    parent_folder = Path(tar_file).parent.parent

                additional_destination_folder = os.path.join(parent_folder, additional_sub_folder)

                if not os.path.exists(additional_destination_folder):
                    os.mkdir(additional_destination_folder)

                os.system(f'cd {current_folder}; tar xf {current_file} --directory {additional_destination_folder}')

            os.remove(tar_file)

    # only keep one LICENSE file

    license_files = sorted(glob.glob(source_folder + '/**/LICENSE', recursive=True))

    for j, license_file in enumerate(license_files):

        current_folder = Path(license_file).parent
        current_file = Path(license_file).name

        if j + 1 == len(license_files):
            os.system(f'cd {current_folder}; mv {current_file} {source_folder}/{current_file}')
        else:
            os.remove(license_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, default=DOWNLOAD_FOLDER)
    parser.add_argument('--extra_destination', type=str, default='all')

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    now = time.time()
    timestamp = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d_%H-%M-%S')

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(filename=f'{args.folder}/{timestamp}_download.log')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    ################
    # Main Dataset #
    ################

    # num_tar_files = {'training': 32,
    #                  'validation': 8,
    #                  'testing': 8}
    #
    # for split in ['training', 'validation', 'testing']:
    #
    #     for i in range(num_tar_files[split]):
    #
    #         file = f'{split}/{split}_%04d.tar' % i
    #         folder = f'{args.folder}/{split}'
    #         url_template = f'gs://waymo_open_dataset_v_1_2_0/{file}'
    #
    #         if not os.path.exists(folder):
    #             os.mkdir(folder)
    #
    #         flag = os.system(f'gsutil cp {url_template} {folder}')
    #
    #         if flag == 0:
    #             logger.info(f'{file} successfully downloaded.')
    #         else:
    #             logger.error(f'{file} failed.')

    ########################
    # Domain Adaption Part #
    ########################

    num_tar_files = {'training': 4,
                     'validation': 2,
                     'testing': 5}

    for split in ['training', 'validation', 'testing']:

        for i in range(num_tar_files[split]):

            file = f'{split}/{split}_%04d.tar' % i
            folder = f'{args.folder}/domain_adaptation/{split}'
            url_template = f'gs://waymo_open_dataset_v_1_2_0/domain_adaptation/{file}'

            if not os.path.exists(folder):
                os.mkdir(folder)

            flag = os.system(f'{gsutil} cp {url_template} {folder}')

            if flag == 0:
                logger.info(f'{file} successfully downloaded.')
            else:
                logger.error(f'{file} failed.')

        # download unlabeled part

        if split  in ['training', 'validation']:

            num_tar_files_unlabeled = {'training': 24,
                                       'validation': 6}

            for i in range(num_tar_files_unlabeled[split]):

                file = f'{split}/unlabeled/{split}_%04d.tar' % i
                folder = f'{args.folder}/domain_adaptation/{split}/unlabeled'
                url_template = f'gs://waymo_open_dataset_v_1_2_0/domain_adaptation/{file}'

                if not os.path.exists(folder):
                    os.mkdir(folder)

                flag = os.system(f'{gsutil} cp {url_template} {folder}')

                if flag == 0:
                    logger.info(f'{file} successfully downloaded.')
                else:
                    logger.error(f'{file} failed.')

    extract_all_tar_files(source_folder=args.folder, additional_sub_folder=args.extra_destination)