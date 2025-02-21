
from torchsig.datasets.wideband import NewWideband, StaticWideband
from torchsig.datasets.narrowband import NewNarrowband, StaticNarrowband
import yaml
import numpy as np
from torchsig.utils.writer import DatasetCreator

yaml_filename = "yaml_test"
wideband_filename = f"{yaml_filename}_wideband"
narrowband_filename = f"{yaml_filename}_narrowband"

def compare(s1, s2):
    d1, t1 = s1
    d2, t2 = s2

    data_matches = np.all(d1 == d2)
    targets_match = t1 == t2

    return data_matches and targets_match

def wideband():
    yaml_file = f"{wideband_filename}.yaml"

    WB = NewWideband(yaml_file)
    # print(WB)

    test_idx = np.random.randint(len(WB))

    dc = DatasetCreator(
        WB,
        wideband_filename,
        overwrite=True
    )

    dc.create()

    WBS = StaticWideband(
        root = wideband_filename,
        impaired = False,
        raw = False,
    )

    WBS2 = StaticWideband(
        root = wideband_filename,
        impaired = False,
        raw = False,
    )

    match = compare(WBS[test_idx], WBS2[test_idx])
    if not match:
        print("Does not match.")
        breakpoint()
    print("Success.")

def narrowband():
    yaml_file = f"{narrowband_filename}.yaml"

    NB = NewNarrowband(yaml_file)
    # print(NB)

    test_idx = np.random.randint(len(NB))

    dc = DatasetCreator(
        NB,
        narrowband_filename,
        overwrite=True
    )

    dc.create()

    NBS = StaticNarrowband(
        root = narrowband_filename,
        impaired = True,
        raw = True,
    )

    # print(NBS)

    NBS2 = StaticNarrowband(
        root = narrowband_filename,
        impaired = True,
        raw = True,
    )

    match = compare(NBS[test_idx], NBS2[test_idx])
    if not match:
        print("Does not match.")
        breakpoint()
    print("Success.")

if __name__=='__main__':
    narrowband()
    wideband()
    

    # breakpoint()

