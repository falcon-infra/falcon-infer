import pytest

from vllm.distributed.eplb.ascend.core.policy.policy_abstract import DynamicConfig
from vllm.distributed.eplb.ascend.core.policy.policy_default_eplb import DefaultEplb
from vllm.distributed.eplb.ascend.core.policy.policy_swift_balancer import SwiftBalanceEplb
from vllm.distributed.eplb.ascend.core.policy.policy_factory import PolicyFactory
from vllm.distributed.eplb.ascend.core.policy.policy_random import RandomLoadBalance


@pytest.fixture
def dummy_config():
    return DynamicConfig()


@pytest.mark.parametrize("policy_type, expected_class", [
    (0, RandomLoadBalance),
    (1, DefaultEplb),
    (2, SwiftBalanceEplb),
    (999, RandomLoadBalance),
])
def test_generate_policy(policy_type, expected_class, dummy_config):
    policy_instance = PolicyFactory.generate_policy(policy_type, dummy_config)
    assert isinstance(policy_instance, expected_class)
