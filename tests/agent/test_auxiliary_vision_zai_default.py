"""_PROVIDER_VISION_MODELS defaults.

The Z.AI entry was hardcoded to ``glm-5v-turbo``; that model is gated
behind a separate subscription and is not reachable on any Z.AI
coding-plan tier (Lite/Pro/Pro+/Max).  Every coding-plan user got 1311
on the first vision call (#46050).
"""

import pytest

from agent.auxiliary_client import _PROVIDER_VISION_MODELS


class TestProviderVisionModels:
    def test_zai_default_is_glm_4_5v(self):
        assert _PROVIDER_VISION_MODELS["zai"] == "glm-4.5v"

    def test_zai_default_not_gated_model(self):
        assert "5v-turbo" not in _PROVIDER_VISION_MODELS["zai"]

    def test_xiaomi_default_unchanged(self):
        assert _PROVIDER_VISION_MODELS["xiaomi"] == "mimo-v2.5"


@pytest.mark.parametrize(
    "provider,model",
    [("zai", "glm-4.5v"), ("xiaomi", "mimo-v2.5")],
)
def test_vision_default_table_shape(provider, model):
    assert _PROVIDER_VISION_MODELS[provider] == model
