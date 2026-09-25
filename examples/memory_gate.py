"""The pattern hermes-sophia uses: similarity finds candidates, a readout decides whether they answer the message."""
import os

from jevless import Decider, LMStudio

decide = Decider(LMStudio(os.environ.get("JEVLESS_MODEL", "qwen/qwen3.5-9b")))
state = {"message": "When did Voyager 2 launch?",
         "memories": ["Voyager 1 launched on September 5, 1977.", "Voyager 1 entered interstellar space in 2012."]}
a = decide.noul(state, "At least one memory directly answers the message.")
print(f"inject={a.noul >= 0.5} p={a.noul:.2f} flip={a.flip}")        # near-miss: similar, but not an answer
