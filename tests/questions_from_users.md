# User Questions Log

Real questions asked during testing, captured to expand test coverage.
Add new questions here as they come up, then periodically fold them into
`test_question_classifier.py`.

## Format

Each entry: the question, what category it got, whether the answer was good,
and any notes on what could be improved.

---

### 2026-03-01

| # | Question | Got | Good answer? | Notes |
|---|----------|-----|-------------|-------|
| 1 | What hearings have occurred in the case? | keyword_listing | Bad initially (literal grep found nothing), good after two-stage semantic search | Led to keyword_listing routing change |
| 2 | How many creditors are there in the case? | keyword_listing | Decent — correctly said it couldn't give a count, pointed to schedules/claims register | Answer quality limited by 7% doc coverage, not routing |
