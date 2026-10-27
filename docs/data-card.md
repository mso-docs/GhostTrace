# Synthetic-v1 data card

Thirty original English synthetic scenarios cover accidental breakage, missed commitments, credit-taking, avoidable inconvenience and careless communication, six per family. No scenario describes an identifiable person or graphic content. AI-assisted text is released under CC0-1.0; see datasets/synthetic-v1/LICENSE.

The first three families form the 18-scenario development partition. The remaining two form the 12-scenario holdout. Family membership is the split unit; paraphrases, seeds and continuations do not create independent examples. Holdout is small and cannot support population-wide conclusions.

Scenarios request two sentences from the fictional actor. Each has the same neutral continuation. Its exact token count and first-16 token mask must be verified using the pinned tokenizer. All current review_status fields are pending_human_review: do not claim human review or run a confirmatory pilot until that review is complete.

Manifest file hashes identify exact raw bytes. Record sha256 hashes use sorted keys, compact separators, UTF-8, ensure_ascii=False, excluding the sha256 field itself. Changes create a new dataset version and new preregistration hash. Never alter scenarios to improve separation after viewing results.
