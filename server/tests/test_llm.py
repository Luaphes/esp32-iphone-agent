import asyncio
import json

import httpx
import pytest

from hidagent.llm import LLM, LLMResponseError


def fake_server(llm: LLM, wants: str) -> list[dict]:
    """Point the client at an in-memory endpoint that only accepts the `wants` token parameter."""
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if wants not in body:
            return httpx.Response(400, json={"error": {"message": f"Unsupported parameter. Use '{wants}' instead."}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "DO: wait"}}]})

    llm._http = httpx.AsyncClient(base_url="http://test/v1", transport=httpx.MockTransport(handle))
    return seen


def test_classic_servers_get_max_tokens_once():
    llm = LLM("http://test/v1", "qwen")
    seen = fake_server(llm, "max_tokens")
    assert asyncio.run(llm.chat("sys", "hi")) == "DO: wait"
    assert len(seen) == 1 and seen[0]["max_tokens"] == 600


def test_auto_switches_to_max_completion_tokens_and_remembers():
    llm = LLM("http://test/v1", "gpt")
    seen = fake_server(llm, "max_completion_tokens")
    asyncio.run(llm.chat("sys", "hi"))
    asyncio.run(llm.chat("sys", "again"))
    assert [("max_tokens" in b, "max_completion_tokens" in b) for b in seen] == [(True, False), (False, True),
                                                                                 (False, True)]


def test_explicit_token_param_is_used_directly():
    llm = LLM("http://test/v1", "gpt", token_param="max_completion_tokens")
    seen = fake_server(llm, "max_completion_tokens")
    asyncio.run(llm.chat("sys", "hi"))
    assert len(seen) == 1


def test_extra_body_and_image_are_sent():
    llm = LLM("http://test/v1", "gpt", extra={"reasoning_effort": "none"})
    seen = fake_server(llm, "max_tokens")
    asyncio.run(llm.chat("sys", "hi", image=b"\xff\xd8jpeg"))
    body = seen[0]
    assert body["reasoning_effort"] == "none"
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def response_server(llm: LLM, response: dict):
    llm._http = httpx.AsyncClient(
        base_url="http://test/v1",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
    )


@pytest.mark.parametrize("content", ["DO: done 已安装", "DO: tap 628 48", "THINK: still working"])
def test_truncated_reply_is_rejected_even_when_it_contains_a_valid_action(content):
    llm = LLM("http://test/v1", "vision")
    response_server(llm, {"choices": [{"finish_reason": "length", "message": {"content": content}}]})
    with pytest.raises(LLMResponseError, match="truncated.*finish_reason=length.*discarded"):
        asyncio.run(llm.chat("sys", "install"))


@pytest.mark.parametrize(("response", "diagnostic"), [
    ({"choices": []}, r"choices\[0\]"),
    ({}, r"choices\[0\]"),
    ({"choices": [None]}, r"choices\[0\]"),
    ({"choices": [{}]}, r"missing choices\[0\].message"),
    ({"choices": [{"message": {}}]}, "no usable text"),
    ({"choices": [{"message": {"content": None}}]}, "no usable text"),
    ({"choices": [{"message": {"content": "  "}}]}, "no usable text"),
    ({"choices": [{"message": {"content": [{"type": "text", "text": "DO: done"}]}}]}, "no usable text"),
])
def test_unusable_response_has_a_clear_diagnostic(response, diagnostic):
    llm = LLM("http://test/v1", "vision")
    response_server(llm, response)
    with pytest.raises(LLMResponseError, match=diagnostic):
        asyncio.run(llm.chat("sys", "hi"))


def test_complete_reply_preserves_content():
    llm = LLM("http://test/v1", "vision")
    response_server(llm, {"choices": [{"finish_reason": "stop", "message": {"content": "THINK: ready\nDO: wait"}}]})
    assert asyncio.run(llm.chat("sys", "hi")) == "THINK: ready\nDO: wait"


@pytest.mark.parametrize("reason", ["content_filter", "tool_calls", "function_call", "error"])
def test_nonfinal_response_cannot_supply_actions(reason):
    llm = LLM("http://test/v1", "vision")
    response_server(llm, {"choices": [{"finish_reason": reason, "message": {"content": "DO: done"}}]})
    with pytest.raises(LLMResponseError, match="unusable finish_reason"):
        asyncio.run(llm.chat("sys", "hi"))
