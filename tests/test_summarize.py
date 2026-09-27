from unittest.mock import MagicMock, patch

from memory.conversation_db import _summarize_with_llm


OLLAMA_CFG = {
    "summarization_provider": "ollama",
    "summarization_model": "gemma4:31b-cloud",
    "summarization_url": "http://localhost:11434",
    "summarization_api_key": "",
}


class TestOllamaSummarization:
    def test_ollama_path_returns_content(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "KEY POINTS: user likes Python"}}
        with patch("memory.conversation_db._get_config", return_value=OLLAMA_CFG), \
             patch("requests.post", return_value=resp) as mock_post:
            result = _summarize_with_llm("some long conversation text")
        assert result == "KEY POINTS: user likes Python"
        assert mock_post.call_args.args[0] == "http://localhost:11434/api/chat"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "gemma4:31b-cloud"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.3
        assert payload["messages"][0] == {"role": "system", "content": "You are a conversation summarizer."}
        assert payload["messages"][1]["role"] == "user"

    def test_ollama_fails_falls_back_to_main_llm(self):
        with patch("memory.conversation_db._get_config", return_value=OLLAMA_CFG), \
             patch("requests.post", side_effect=RuntimeError("conn refused")), \
             patch("core.llm_client.call_llm_text", return_value="FALLBACK SUMMARY") as mock_fb:
            result = _summarize_with_llm("some long conversation text")
        assert result == "FALLBACK SUMMARY"
        mock_fb.assert_called_once()

    def test_ollama_fails_and_main_fails_returns_none(self):
        with patch("memory.conversation_db._get_config", return_value=OLLAMA_CFG), \
             patch("requests.post", side_effect=RuntimeError("conn refused")), \
             patch("core.llm_client.call_llm_text", side_effect=RuntimeError("boom")):
            result = _summarize_with_llm("some long conversation text")
        assert result is None


class TestGeminiSummarization:
    def test_gemini_path_returns_content(self):
        cfg = {
            "summarization_provider": "gemini",
            "summarization_model": "gemini-2.5-flash",
            "summarization_api_key": "test-key",
        }
        fake_result = MagicMock()
        fake_result.text = "GEMINI SUMMARY"
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_result
        with patch("memory.conversation_db._get_config", return_value=cfg), \
             patch("google.genai.Client", return_value=fake_client) as mock_client:
            result = _summarize_with_llm("some long conversation text")
        assert result == "GEMINI SUMMARY"
        mock_client.assert_called_once_with(api_key="test-key")

    def test_gemini_without_key_falls_back_to_main_llm(self):
        cfg = {
            "summarization_provider": "gemini",
            "summarization_model": "gemini-2.5-flash",
            "summarization_api_key": "",
        }
        with patch("memory.conversation_db._get_config", return_value=cfg), \
             patch("core.llm_client.call_llm_text", return_value="MAIN SUMMARY") as mock_fb:
            result = _summarize_with_llm("some long conversation text")
        assert result == "MAIN SUMMARY"
        mock_fb.assert_called_once()


class TestAutoMode:
    def test_no_provider_uses_main_llm(self):
        with patch("memory.conversation_db._get_config", return_value={}), \
             patch("core.llm_client.call_llm_text", return_value="MAIN SUMMARY") as mock_fb:
            result = _summarize_with_llm("some long conversation text")
        assert result == "MAIN SUMMARY"
        mock_fb.assert_called_once()

    def test_truncates_input_to_60000_chars(self):
        # 80k chars → chunked into 60k + 20k (no more silent 12k cut).
        long_text = "A" * 80000
        with patch("memory.conversation_db._get_config", return_value={}), \
             patch("core.llm_client.call_llm_text", return_value="S") as mock_fb:
            result = _summarize_with_llm(long_text)
        assert mock_fb.call_count == 2
        first_prompt = mock_fb.call_args_list[0].args[0]
        assert "A" * 60000 in first_prompt
        assert len(first_prompt) <= 60000 + 400
        # Both chunk summaries merged (small result → no final condense call)
        assert result == "S\n\nS"