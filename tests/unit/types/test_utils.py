import pytest

from litellm.litellm_core_utils.initialize_dynamic_callback_params import validate_excluded_services_value
from litellm.types.utils import (
    CAPTURE_MESSAGE_CONTENT_VALUES,
    CaptureMessageContent,
    PromptTokensDetailsWrapper,
    captures_span_content,
)


@pytest.mark.parametrize("short,long", [(4, 5), (0, 0)])
def test_native_cache_write_lifetimes_normalize_without_losing_wire_fields(short: int, long: int) -> None:
    source = {"ttl_30m_tokens": short, "ttl_12h_tokens": long}
    details = PromptTokensDetailsWrapper(cache_write_tokens_details=source)
    wire = details.model_dump()
    assert wire["cache_write_tokens_details"] == source
    assert wire["cache_creation_token_details"]["ephemeral_30m_input_tokens"] == short
    assert wire["cache_creation_token_details"]["ephemeral_12h_input_tokens"] == long
    assert source == {"ttl_30m_tokens": short, "ttl_12h_tokens": long}


@pytest.mark.parametrize("value", ["", "  ", "auth", "none,redis", "redis,", ",postgres"])
def test_invalid_excluded_services_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=r"Invalid excluded_services.*none"):
        validate_excluded_services_value(value)


@pytest.mark.parametrize(
    "mode",
    [
        CaptureMessageContent.NO_CONTENT,
        CaptureMessageContent.SPAN_ONLY,
        CaptureMessageContent.EVENT_ONLY,
        CaptureMessageContent.SPAN_AND_EVENT,
    ],
)
def test_capture_message_content_members_are_plain_strings(mode: str) -> None:
    assert type(mode) is str
    assert (str(mode), f"{mode}", repr(mode)) == (mode, mode, repr(str(mode)))


def test_a_destination_may_only_set_the_span_capture_modes() -> None:
    assert CAPTURE_MESSAGE_CONTENT_VALUES == {"no_content", "span_only"}


def test_capture_message_content_constructor_accepts_any_string() -> None:
    assert CaptureMessageContent("custom") == "custom"


@pytest.mark.parametrize(
    ("mode", "captures"),
    [
        ("no_content", False),
        ("span_only", True),
        ("event_only", False),
        ("span_and_event", True),
        ("bogus", False),
        (None, False),
    ],
)
def test_captures_span_content(mode: str | None, captures: bool) -> None:
    assert captures_span_content(mode) is captures
