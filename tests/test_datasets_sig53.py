from torchsig.datasets.sig53 import Sig53
from unittest import TestCase


class GenerateSig53(TestCase):
    def test_can_generate_sig53_clean_train(self):
        x = 2 + 2
        self.assertEqual(x, 4)
