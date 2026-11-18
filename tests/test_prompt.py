from video_analysis.semantic.prompt import SYSTEM_INSTRUCTION, build_semantic_prompt


def test_prompt_enforces_separate_semantic_artifact():
    assert "Do not reproduce the input JSON" in SYSTEM_INSTRUCTION
    prompt = build_semantic_prompt(9.568)
    assert "9.568 seconds" in prompt
