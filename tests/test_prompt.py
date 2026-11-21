from video_analysis.semantic.prompt import SYSTEM_INSTRUCTION, build_semantic_prompt


def test_prompt_enforces_separate_semantic_artifact():
    assert "Do not reproduce the input JSON" in SYSTEM_INSTRUCTION
    prompt = build_semantic_prompt(9.568)
    assert "9.568 seconds" in prompt


def test_prompt_keeps_core_guidance_and_video_type():
    for phrase in ["audio_observations", "INTERACTIONS", "EDITING OPPORTUNITIES",
                   "VIDEO TYPE", "screen_recording", "Do not reproduce the input JSON"]:
        assert phrase in SYSTEM_INSTRUCTION


def test_prompt_screen_actions_in_same_call():
    assert "screen_actions" in SYSTEM_INSTRUCTION
    assert "Never invent labels" in SYSTEM_INSTRUCTION
    assert "empty array" in SYSTEM_INSTRUCTION
    assert "screen_actions" in build_semantic_prompt(5.0)
