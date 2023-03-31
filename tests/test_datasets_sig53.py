from torchsig.datasets.sig53 import Sig53
from unittest import TestCase


class GenerateSig53(TestCase):
    def test_can_generate_sig53_clean_train(self):
        ds = Sig53(root="tests/", impaired=False, regenerate=True, generation_test=True)
        first_data = ds[0][0]
        ds = Sig53(root="tests/", impaired=False, regenerate=True, generation_test=True)
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())

    def test_can_generate_sig53_clean_val(self):
        ds = Sig53(
            root="tests/",
            impaired=False,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        first_data = ds[0][0]
        ds = Sig53(
            root="tests/",
            impaired=False,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())

    def test_can_generate_sig53_impaired_train(self):
        ds = Sig53(root="tests/", impaired=True, regenerate=True, generation_test=True)
        first_data = ds[0][0]
        ds = Sig53(root="tests/", impaired=True, regenerate=True, generation_test=True)
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())

    def test_can_generate_sig53_impaired_val(self):
        ds = Sig53(
            root="tests/",
            impaired=True,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        first_data = ds[0][0]
        ds = Sig53(
            root="tests/",
            impaired=True,
            train=False,
            regenerate=True,
            generation_test=True,
        )
        second_data = ds[0][0]

        self.assertEqual(first_data.all(), second_data.all())
