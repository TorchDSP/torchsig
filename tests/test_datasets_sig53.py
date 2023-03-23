from torchsig.datasets.sig53 import Sig53
from unittest import TestCase


class GenerateSig53(TestCase):
    def can_generate_sig53_clean_train(self):
        Sig53(root=".", train=True, impaired=False)
        return True
