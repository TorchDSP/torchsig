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
class Sig53GenerationTestConfig(Sig53Config):
    name: str = "sig53_generation_test"
    seed: int = 1234567890
    eb_no: bool = False
    num_samples: int = 106
    level: int = 2


@dataclass
class Sig53CleanTrainConfig(Sig53Config):
    name: str = "sig53_clean_train"
    seed: int = 1234567890
    eb_no: bool = False
    num_samples: int = 1_060_000
    level: int = 0


@dataclass
class Sig53CleanValConfig(Sig53CleanTrainConfig):
    name: str = "sig53_clean_val"
    seed: int = 1234567891
    eb_no: bool = False
    num_samples: int = 106_000


@dataclass
class Sig53ImpairedTrainConfig(Sig53Config):
    name: str = "sig53_impaired_train"
    seed: int = 1234567892
    eb_no: bool = False
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class Sig53ImpairedValConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 106_000


@dataclass
class Sig53ImpairedEbNoTrainConfig(Sig53Config):
    name: str = "sig53_impaired_ebno_train"
    seed: int = 1234567892
    eb_no: bool = True
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class Sig53ImpairedEbNoValConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_ebno_val"
    seed: int = 1234567893
    eb_no: bool = True
    num_samples: int = 106_000


@dataclass
class WidebandSig53Config:
    name: str
    num_samples: int
    level: int
    seed: int
    num_iq_samples: int = int(512 * 512)
    use_gpu: bool = True


@dataclass
class WidebandSig53CleanTrainConfig(WidebandSig53Config):
    name: str = "wideband_sig53_clean_train"
    seed: int = 1234567890
    num_samples: int = 250_000
    level: int = 1


@dataclass
class WidebandSig53CleanValConfig(WidebandSig53CleanTrainConfig):
    name: str = "wideband_sig53_clean_val"
    seed: int = 1234567891
    num_samples: int = 25_000


@dataclass
class WidebandSig53ImpairedTrainConfig(WidebandSig53Config):
    name: str = "wideband_sig53_impaired_train"
    seed: int = 1234567892
    num_samples: int = 250_000
    level: int = 2


@dataclass
class WidebandSig53ImpairedValConfig(WidebandSig53ImpairedTrainConfig):
    name: str = "wideband_sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 25_000
