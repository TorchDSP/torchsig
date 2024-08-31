"""TorchSig Dataset Configs
"""
from dataclasses import dataclass

@dataclass
class Sig53Config:
    name: str
    num_samples: int
    level: int
    seed: int
    eb_no: bool
    num_iq_samples: int = 4096
    use_class_idx: bool = True
    include_snr: bool = True


@dataclass
class Sig53CleanTrainConfig(Sig53Config):
    name: str = "sig53_clean_train"
    seed: int = 1234567890
    eb_no: bool = False
    num_samples: int = 1_060_000
    level: int = 0

@dataclass
class Sig53CleanEbNoTrainConfig(Sig53CleanTrainConfig):
    eb_no: bool = True

@dataclass
class Sig53CleanTrainQAConfig(Sig53CleanTrainConfig):
    num_samples: int = 10_600
    
@dataclass
class Sig53CleanEbNoTrainQAConfig(Sig53CleanTrainQAConfig):
    eb_no: bool = True


@dataclass
class Sig53CleanValConfig(Sig53CleanTrainConfig):
    name: str = "sig53_clean_val"
    seed: int = 1234567891
    eb_no: bool = False
    num_samples: int = 10_600

@dataclass
class Sig53CleanEbNoValConfig(Sig53CleanValConfig):
    eb_no: bool = True

@dataclass
class Sig53CleanValQAConfig(Sig53CleanValConfig):
    num_samples: int = 1060

@dataclass
class Sig53CleanEbNoValQAConfig(Sig53CleanValQAConfig):
    eb_no: bool = True


@dataclass
class Sig53ImpairedTrainConfig(Sig53Config):
    name: str = "sig53_impaired_train"
    seed: int = 1234567892
    eb_no: bool = False
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class Sig53ImpairedTrainQAConfig(Sig53ImpairedTrainConfig):
    num_samples: int = 10_600


@dataclass
class Sig53ImpairedValConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 106_000


@dataclass
class Sig53ImpairedValQAConfig(Sig53ImpairedValConfig):
    num_samples: int = 1060


@dataclass
class Sig53ImpairedEbNoTrainConfig(Sig53Config):
    name: str = "sig53_impaired_ebno_train"
    seed: int = 1234567892
    eb_no: bool = True
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class Sig53ImpairedEbNoTrainQAConfig(Sig53ImpairedEbNoTrainConfig):
    num_samples: int = 10_600


@dataclass
class Sig53ImpairedEbNoValConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_ebno_val"
    seed: int = 1234567893
    eb_no: bool = True
    num_samples: int = 106_000


@dataclass
class Sig53ImpairedEbNoValQAConfig(Sig53ImpairedEbNoValConfig):
    num_samples: int = 1060


@dataclass
class WidebandSig53Config:
    name: str
    num_samples: int
    level: int
    seed: int
    num_iq_samples: int = int(512 * 512)
    overlap_prob: float = 0.0


@dataclass
class WidebandSig53CleanTrainConfig(WidebandSig53Config):
    name: str = "wideband_sig53_clean_train"
    seed: int = 1234567890
    num_samples: int = 250_000
    level: int = 1
    overlap_prob:float = 0.0


@dataclass
class WidebandSig53CleanTrainQAConfig(WidebandSig53CleanTrainConfig):
    num_samples: int = 250


@dataclass
class WidebandSig53CleanValConfig(WidebandSig53CleanTrainConfig):
    name: str = "wideband_sig53_clean_val"
    seed: int = 1234567891
    num_samples: int = 25_000


@dataclass
class WidebandSig53CleanValQAConfig(WidebandSig53CleanValConfig):
    num_samples: int = 250


@dataclass
class WidebandSig53ImpairedTrainConfig(WidebandSig53Config):
    name: str = "wideband_sig53_impaired_train"
    seed: int = 1234567892
    num_samples: int = 250_000
    level: int = 2
    overlap_prob: float = 0.1 #TODO


@dataclass
class WidebandSig53ImpairedTrainQAConfig(WidebandSig53ImpairedTrainConfig):
    num_samples: int = 250


@dataclass
class WidebandSig53ImpairedValConfig(WidebandSig53ImpairedTrainConfig):
    name: str = "wideband_sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 25_000


@dataclass
class WidebandSig53ImpairedValQAConfig(WidebandSig53ImpairedValConfig):
    num_samples: int = 250
