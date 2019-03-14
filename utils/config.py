from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    train_data_dir = '/home/xpwang/Desktop/models/research/my_model/predict_resoult.txt'
    test_data_dir = '/home/xpwang/Desktop/models/research/my_model/predict_resoult_eval.txt'

    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-2

    use_drop = True
    use_adam = False


    # visualization
    env = 'unet-faster-rcnn'  # visdom env
    port = 8097
    plot_every = 4  # vis every N iter



    # training
    epoch = 18





    test_num = 800
    # model
    load_path = None



    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
