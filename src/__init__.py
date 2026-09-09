# module exports for acoustic vehicle speed estimation

from .config import Config
from .data_loader import (
    VS13Dataset,
    get_vs13_datasets,
    load_audio,
    pad_or_crop_audio,
    apply_gain_augmentation,
    apply_additive_noise,
    create_10fold_splits,
)
from .models import (
    Factorized1DNet,
    Factorized1DSENet,
    Factorized1DBlock,
    SEBlock1D,
    SqueezeExcite1D,
    SincConv1d,
    build_model,
    build_se_resnet,
)
from .losses import (
    CauchyLoss,
    SmoothCauchyLoss,
    HuberLoss,
    KinematicAccelerationLoss,
    PhysicsKinematicLoss,
    DomainBoundaryLoss,
    PhysicsInformedLoss,
    CombinedPhysicsLoss,
)
from .train_engine import (
    train_one_epoch,
    evaluate,
    train_fold,
    evaluate_ensemble,
    run_cross_validation,
)
from .utils import (
    set_seed,
    parse_speed_from_filename,
    discover_dataset_files,
    get_all_audio_paths_and_labels,
    compute_rmse,
    compute_mae,
    profile_peak_memory,
)
