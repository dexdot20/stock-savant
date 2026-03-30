import unittest

from services.ai.providers.reflection_prompt_utils import (
    build_reflection_prompt,
    should_inject_reflection,
)


class ReflectionPromptUtilsTests(unittest.TestCase):
    def test_should_inject_reflection_requires_interval_and_memory(self) -> None:
        working_memory = {
            "facts_learned": [{"text": "Revenue increased 12%.", "importance": 7}],
            "unanswered_questions": [],
            "contradictions_found": [],
            "research_milestones": [],
        }

        self.assertFalse(
            should_inject_reflection(working_memory, step=2, interval_steps=3)
        )
        self.assertTrue(
            should_inject_reflection(working_memory, step=3, interval_steps=3)
        )

    def test_build_reflection_prompt_includes_core_sections(self) -> None:
        working_memory = {
            "facts_learned": [
                {"text": "Revenue increased 12%.", "importance": 7},
                {"text": "Net debt fell to 10B TRY.", "importance": 8},
            ],
            "unanswered_questions": ["Is margin expansion sustainable?"],
            "contradictions_found": ["Source A says backlog rose, source B says it fell."],
            "research_milestones": ["Initial screen completed"],
        }

        prompt = build_reflection_prompt(
            working_memory,
            step=6,
            output_language="Turkish",
        )

        self.assertIn("SELF-REFLECTION CHECKPOINT (step 6)", prompt)
        self.assertIn("Verified facts to anchor on:", prompt)
        self.assertIn("Open questions to resolve next:", prompt)
        self.assertIn("Contradictions to verify:", prompt)
        self.assertIn("What single next tool call would most reduce uncertainty?", prompt)


if __name__ == "__main__":
    unittest.main()