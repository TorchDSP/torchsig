"""TorchSig Dataset Configs
"""
from dataclasses import dataclass

@dataclass
class NarrowbandConfig:
    name: str
    num_samples: int
    level: int
    seed: int
    eb_no: bool
    num_iq_samples: int = 4096
    use_class_idx: bool = True
    include_snr: bool = True


@dataclass
class NarrowbandCleanTrainConfig(NarrowbandConfig):
    name: str = "narrowband_clean_train"
    seed: int = 1234567890
    eb_no: bool = False
    num_samples: int = 1_060_000
    level: int = 0

@dataclass
class NarrowbandCleanEbNoTrainConfig(NarrowbandCleanTrainConfig):
    eb_no: bool = True

@dataclass
class NarrowbandCleanTrainQAConfig(NarrowbandCleanTrainConfig):
    num_samples: int = 10_600
    
@dataclass
class NarrowbandCleanEbNoTrainQAConfig(NarrowbandCleanTrainQAConfig):
    eb_no: bool = True


@dataclass
class NarrowbandCleanValConfig(NarrowbandCleanTrainConfig):
    name: str = "narrowband_clean_val"
    seed: int = 1234567891
    eb_no: bool = False
    num_samples: int = 10_600

@dataclass
class NarrowbandCleanEbNoValConfig(NarrowbandCleanValConfig):
    eb_no: bool = True

@dataclass
class NarrowbandCleanValQAConfig(NarrowbandCleanValConfig):
    num_samples: int = 1060

@dataclass
class NarrowbandCleanEbNoValQAConfig(NarrowbandCleanValQAConfig):
    eb_no: bool = True


@dataclass
class NarrowbandImpairedTrainConfig(NarrowbandConfig):
    name: str = "narrowband_impaired_train"
    seed: int = 1234567892
    eb_no: bool = False
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class NarrowbandImpairedTrainQAConfig(NarrowbandImpairedTrainConfig):
    num_samples: int = 10_600


@dataclass
class NarrowbandImpairedValConfig(NarrowbandImpairedTrainConfig):
    name: str = "narrowband_impaired_val"
    seed: int = 1234567893
    num_samples: int = 106_000


@dataclass
class NarrowbandImpairedValQAConfig(NarrowbandImpairedValConfig):
    num_samples: int = 1060


@dataclass
class NarrowbandImpairedEbNoTrainConfig(NarrowbandConfig):
    name: str = "narrowband_impaired_ebno_train"
    seed: int = 1234567892
    eb_no: bool = True
    num_samples: int = 5_300_000
    level: int = 2


@dataclass
class NarrowbandImpairedEbNoTrainQAConfig(NarrowbandImpairedEbNoTrainConfig):
    num_samples: int = 10_600


@dataclass
class NarrowbandImpairedEbNoValConfig(NarrowbandImpairedTrainConfig):
    name: str = "narrowband_impaired_ebno_val"
    seed: int = 1234567893
    eb_no: bool = True
    num_samples: int = 106_000


@dataclass
class NarrowbandImpairedEbNoValQAConfig(NarrowbandImpairedEbNoValConfig):
    num_samples: int = 1060


@dataclass
class WidebandConfig:
    name: str
    num_samples: int
    level: int
    seed: int
    num_iq_samples: int = int(512 * 512)
    overlap_prob: float = 0.0


@dataclass
class WidebandCleanTrainConfig(WidebandConfig):
    name: str = "wideband_clean_train"
    seed: int = 1234567890
    num_samples: int = 250_000
    level: int = 1
    overlap_prob:float = 0.0


@dataclass
class WidebandCleanTrainQAConfig(WidebandCleanTrainConfig):
    num_samples: int = 250


@dataclass
class WidebandCleanValConfig(WidebandCleanTrainConfig):
    name: str = "wideband_clean_val"
    seed: int = 1234567891
    num_samples: int = 25_000


@dataclass
class WidebandCleanValQAConfig(WidebandCleanValConfig):
    num_samples: int = 250


@dataclass
class WidebandImpairedTrainConfig(WidebandConfig):
    name: str = "wideband_impaired_train"
    seed: int = 1234567892
    num_samples: int = 250_000
    level: int = 2
    overlap_prob: float = 0.1


@dataclass
class WidebandImpairedTrainQAConfig(WidebandImpairedTrainConfig):
    num_samples: int = 250


@dataclass
class WidebandImpairedValConfig(WidebandImpairedTrainConfig):
    name: str = "wideband_impaired_val"
    seed: int = 1234567893
    num_samples: int = 25_000


@dataclass
class WidebandImpairedValQAConfig(WidebandImpairedValConfig):
    num_samples: int = 250
