import ml_collections  # type: ignore


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 32
    config.epochs = 10
    config.mask_ratio = 0.5
    config.start_TS_epoch = 5
    config.masking_method = "random_masking"
    config.output_dir = "experiments/avenue"  # the checkpoints will be loaded from here
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = False
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4
    config.warmup_epochs = 2
    config.min_lr = 1e-6
    config.clip_grad = 0.05

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = r"C:\Users\Anay\.gemini\antigravity\scratch\vad\Avenue_Extracted\Avenue Dataset"
    config.avenue_gt_path = r"C:\Users\Anay\.gemini\antigravity\scratch\vad\data\avenue\gt_txt_labels"
    config.percent_abnormal = 0.0
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 8
    config.pin_mem = True

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.epochs = 200
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "experiments/shanghai" # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L1'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "train"
    config.resume=False

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "/media/alin/hdd/SanhaiTech"
    config.shanghai_gt_path = "/media/alin/hdd/Transformer_Labels/Shanghai_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config
