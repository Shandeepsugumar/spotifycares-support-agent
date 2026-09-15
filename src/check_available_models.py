"""
Model availability check.

Different Groq accounts/keys can have access to different models (some are
gated, some are region-limited, some get deprecated). Rather than guessing a
model name from documentation, this asks Groq's API directly what THIS key
can actually use, then runs a tiny smoke-test chat call against the first
reasonable candidate to confirm it really works end-to-end.
"""
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("Fetching available models for this API key...")
models = client.models.list()

print(f"\n{len(models.data)} models available:\n")
for m in models.data:
    print(f"  - {m.id}")

# Try to find a good candidate: prefer larger-context instruct models that
# aren't whisper/tts/guard/moderation models, since we need a chat model
# with decent free-tier TPM.
candidates = [
    m.id for m in models.data
    if "whisper" not in m.id.lower()
    and "tts" not in m.id.lower()
    and "guard" not in m.id.lower()
    and "prompt-guard" not in m.id.lower()
]

print(f"\nLikely usable chat models: {candidates}")

if candidates:
    test_model = candidates[0]
    print(f"\nSmoke-testing '{test_model}' with a trivial chat call...")
    try:
        resp = client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        print(f"SUCCESS. Model responded: {resp.choices[0].message.content!r}")
        print(f"\n==> Use MODEL = \"{test_model}\" in agent.py")
    except Exception as e:
        print(f"FAILED even on this candidate: {e}")
        print("Try the next model in the candidates list above manually.")
else:
    print("No obvious chat-model candidates found -- inspect the full list above manually.")
