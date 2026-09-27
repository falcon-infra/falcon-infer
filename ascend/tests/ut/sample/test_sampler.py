from tests.ut.base import TestBase
from vllm.v1.sample.ascend.sampler import AscendSampler
from vllm.v1.sample.ascend.sampler import AscendTopKTopPSampler


class TestAscendSampler(TestBase):

    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, 'topk_topp_sampler'))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)
