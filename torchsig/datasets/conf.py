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
class Sig53CleanTrainMediumConfig(Sig53Config):
    name: str = "sig53_clean_train"
    seed: int = 1234567890
    eb_no: bool = False
    num_samples: int = 106000
    level: int = 0


@dataclass
class Sig53CleanTrainSmallConfig(Sig53Config):
    name: str = "sig53_clean_train"
    seed: int = 1234567890
    eb_no: bool = False
    num_samples: int = 10600
    level: int = 0


@dataclass
class Sig53CleanTrainQAConfig(Sig53CleanTrainConfig):
    num_samples: int = 106


@dataclass
class Sig53CleanValConfig(Sig53CleanTrainConfig):
    name: str = "sig53_clean_val"
    seed: int = 1234567891
    eb_no: bool = False
    num_samples: int = 106_000


@dataclass
class Sig53CleanValMediumConfig(Sig53CleanTrainConfig):
    name: str = "sig53_clean_val"
    seed: int = 1234567891
    eb_no: bool = False
    num_samples: int = 10600


@dataclass
class Sig53CleanValSmallConfig(Sig53CleanTrainConfig):
    name: str = "sig53_clean_val"
    seed: int = 1234567891
    eb_no: bool = False
    num_samples: int = 1060


@dataclass
class Sig53CleanValQAConfig(Sig53CleanValConfig):
    num_samples: int = 106


@dataclass
class Sig53ImpairedTrainConfig(Sig53Config):
    name: str = "sig53_impaired_train"
    seed: int = 1234567892
    eb_no: bool = False
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class Sig53ImpairedTrainMediumConfig(Sig53Config):
    name: str = "sig53_impaired_train"
    seed: int = 1234567892
    eb_no: bool = False
    num_samples: int = 530000
    level: int = 2


@dataclass
class Sig53ImpairedTrainSmallConfig(Sig53Config):
    name: str = "sig53_impaired_train"
    seed: int = 1234567892
    eb_no: bool = False
    num_samples: int = 53000
    level: int = 2


@dataclass
class Sig53ImpairedTrainQAConfig(Sig53ImpairedTrainConfig):
    num_samples: int = 106


@dataclass
class Sig53ImpairedValConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 106_000


@dataclass
class Sig53ImpairedValMediumConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 10600


@dataclass
class Sig53ImpairedValSmallConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 1060


@dataclass
class Sig53ImpairedValQAConfig(Sig53ImpairedValConfig):
    num_samples: int = 106


@dataclass
class Sig53ImpairedEbNoTrainConfig(Sig53Config):
    name: str = "sig53_impaired_ebno_train"
    seed: int = 1234567892
    eb_no: bool = True
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class Sig53ImpairedEbNoTrainQAConfig(Sig53ImpairedEbNoTrainConfig):
    num_samples: int = 106


@dataclass
class Sig53ImpairedEbNoValConfig(Sig53ImpairedTrainConfig):
    name: str = "sig53_impaired_ebno_val"
    seed: int = 1234567893
    eb_no: bool = True
    num_samples: int = 106_000


@dataclass
class Sig53ImpairedEbNoValQAConfig(Sig53ImpairedEbNoValConfig):
    num_samples: int = 106


@dataclass
class WidebandSig53Config:
    name: str
    num_samples: int
    level: int
    seed: int
    num_iq_samples: int = int(512 * 512)


@dataclass
class WidebandSig53CleanTrainConfig(WidebandSig53Config):
    name: str = "wideband_sig53_clean_train"
    seed: int = 1234567890
    num_samples: int = 250_000
    level: int = 1


@dataclass
class WidebandSig53CleanTrainMediumConfig(WidebandSig53Config):
    name: str = "wideband_sig53_clean_train"
    seed: int = 1234567890
    num_samples: int = 250_00
    level: int = 1


@dataclass
class WidebandSig53CleanTrainSmallConfig(WidebandSig53Config):
    name: str = "wideband_sig53_clean_train"
    seed: int = 1234567890
    num_samples: int = 250_0
    level: int = 1


@dataclass
class WidebandSig53CleanTrainQAConfig(WidebandSig53CleanTrainConfig):
    num_samples: int = 250


@dataclass
class WidebandSig53CleanValConfig(WidebandSig53CleanTrainConfig):
    name: str = "wideband_sig53_clean_val"
    seed: int = 1234567891
    num_samples: int = 25_000


@dataclass
class WidebandSig53CleanValMediumConfig(WidebandSig53CleanTrainConfig):
    name: str = "wideband_sig53_clean_val"
    seed: int = 1234567891
    num_samples: int = 2500


@dataclass
class WidebandSig53CleanValSmallConfig(WidebandSig53CleanTrainConfig):
    name: str = "wideband_sig53_clean_val"
    seed: int = 1234567891
    num_samples: int = 250


@dataclass
class WidebandSig53CleanValQAConfig(WidebandSig53CleanValConfig):
    num_samples: int = 250


@dataclass
class WidebandSig53ImpairedTrainConfig(WidebandSig53Config):
    name: str = "wideband_sig53_impaired_train"
    seed: int = 1234567892
    num_samples: int = 250_000
    level: int = 2


@dataclass
class WidebandSig53ImpairedTrainMediumConfig(WidebandSig53Config):
    name: str = "wideband_sig53_impaired_train"
    seed: int = 1234567892
    num_samples: int = 25000
    level: int = 2


@dataclass
class WidebandSig53ImpairedTrainSmallConfig(WidebandSig53Config):
    name: str = "wideband_sig53_impaired_train"
    seed: int = 1234567892
    num_samples: int = 2500
    level: int = 2


@dataclass
class WidebandSig53ImpairedTrainQAConfig(WidebandSig53ImpairedTrainConfig):
    num_samples: int = 250


@dataclass
class WidebandSig53ImpairedValConfig(WidebandSig53ImpairedTrainConfig):
    name: str = "wideband_sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 25_000


@dataclass
class WidebandSig53ImpairedValMediumConfig(WidebandSig53ImpairedTrainConfig):
    name: str = "wideband_sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 2500


@dataclass
class WidebandSig53ImpairedValSmallConfig(WidebandSig53ImpairedTrainConfig):
    name: str = "wideband_sig53_impaired_val"
    seed: int = 1234567893
    num_samples: int = 250


@dataclass
class WidebandSig53ImpairedValQAConfig(WidebandSig53ImpairedValConfig):
    num_samples: int = 250
