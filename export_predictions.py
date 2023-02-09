import numpy as np
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from pprint import pformat

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
# from hfnet.models import get_model  # noqa: E402
from symbols.get_model import get_sym
from Unsuper.dataset import build_dataloader
# from Unsuper.dataset import get_dataset  # noqa: E402
from Unsuper.utils import tools  # noqa: E402
from Unsuper.settings import EXPER_PATH, DATA_PATH  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    parser.add_argument('--keys', type=str, default='*')
    parser.add_argument('--exper_name', type=str)
    parser.add_argument('--as_dataset', action='store_true')
    args = parser.parse_args()

    export_name = args.export_name
    exper_name = args.exper_name
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    keys = '*' if args.keys == '*' else args.keys.split(',')

    if args.as_dataset:
        base_dir = Path(DATA_PATH, export_name)
    else:
        base_dir = Path(EXPER_PATH, 'exports')
        base_dir = Path(base_dir, ((exper_name+'/') if exper_name else '') + export_name)
    base_dir.mkdir(parents=True, exist_ok=True)

    if exper_name:
        # Update only the model config (not the dataset)
        with open(Path(EXPER_PATH, exper_name, 'config.yaml'), 'r') as f:
            config['model'] = tools.dict_update(
                yaml.safe_load(f)['model'], config.get('model', {}))
        checkpoint_path = Path(EXPER_PATH, exper_name)
        if config.get('weights', None):
            checkpoint_path = Path(checkpoint_path, config['weights'])
    else:
        if config.get('weights', None):
            checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])
        else:
            checkpoint_path = None
            logging.info('No weights provided.')
    logging.info(f'Starting export with configuration:\n{pformat(config)}')

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=config['data'],
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = get_sym(model_config=config['MODEL'], image_shape=config['data']['IMAGE_SHAPE'], is_training=False)

    for data in tqdm(test_set):
        #print(DATA_PATH + '/' + config['data']['name'] + '/' + data['name'].decode('UTF-8') + '.ppm')
        im = cv2.imread(DATA_PATH + '/' + config['data']['name'] + '/' + data['name'].decode('UTF-8') + '.ppm')
        predictions = model.predict(im)
        predictions['input_shape'] = data['image'].shape
        name = data['name'].decode('utf-8')
        Path(base_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
        np.savez(Path(base_dir, '{}.npz'.format(name)), **predictions)


#with get_model(config['model']['name'])(data_shape={'image': [None, None, None, config['model']['image_channels']]},
#            **config['model']) as net:
##            net.load(str(checkpoint_path))
#        dataset = get_dataset(config['data']['name'])(**config['data'])
#        print(dataset)
#        test_set = dataset.get_test_set()
#
#        for data in tqdm(test_set):
#            predictions = net.predict(data, keys=keys)
#            predictions['input_shape'] = data['image'].shape
#            name = data['name'].decode('utf-8')
#            Path(base_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
#            np.savez(Path(base_dir, '{}.npz'.format(name)), **predictions)
