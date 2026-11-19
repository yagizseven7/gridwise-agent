# n8n: email → draft → human approval → reply

`email_approval_workflow.json` turns an email arriving in a support mailbox into a
reply that a person has approved. GridWise drafts the reply; n8n handles mail and
the waiting. Nothing goes out to the customer without a decision.

```
IMAP: New support email
  └─► HTTP POST /email/draft          GridWise strips quoted history, redacts personal
        │                             data, runs the agent, stores a pending draft
        └─► Email the reviewer        draft + confidence + flags + sources + approve/reject links
              └─► Wait (≤ 48 h)        resumes when a link is clicked
                    └─► HTTP POST /drafts/{id}/decision    recorded in GridWise (409 if already decided)
                          └─► If status == approved
                                ├─ yes ─► Email the customer the approved text (Re: subject)
                                └─ no  ─► stop; nothing is sent (rejected, or expired after 48 h)
```

Design choices worth knowing:

- **GridWise never sends email.** It only drafts and records decisions. The reply text
  n8n sends is `final_reply` from the decision call, i.e. exactly what was approved.
- **The sender's address is not stored by GridWise.** n8n keeps it inside the paused
  execution so it can reply; the API only sees the redacted body.
- **Every draft has a one-time approval token.** GridWise issues it with the draft,
  stores only its hash, and refuses (403) any decision without it. The reviewer's
  links carry it, so knowing the draft id or guessing n8n's resume URL is not enough
  (n8n 1.x numbers its resume URLs sequentially).
- **Expiry is a rejection.** If nobody decides in 48 hours, the Wait node times out,
  the decision is recorded as not approved, and nothing is sent.
- **Drafts the agent was unsure about are marked.** `requires_attention` is true when
  confidence is low or any flag was raised; the reviewer's subject line says so.

## Set up

1. `docker compose --profile email up --build` (starts GridWise on :8000 and n8n on :5678).
2. In n8n (http://localhost:5678): **Workflows → Import from file** →
   `email_approval_workflow.json`. Inside the compose network the GridWise URL is
   `http://gridwise:8000`; regenerate for another URL with
   `python scripts/build_n8n_workflow.py --gridwise-url https://your-host`.
3. Create two credentials and select them on the nodes: **IMAP** (support mailbox) on
   *New support email*, **SMTP** on *Ask reviewer to approve* and *Send approved reply*.
4. Change the reviewer address (`duty-energy-manager@example.com`) and sender addresses.
5. Activate (publish) the workflow.

The links are bearer secrets: anyone holding one can decide on that draft. For
production, also turn on authentication for the Wait node's resume webhook
(*Wait for decision → Authentication*) and keep n8n behind your SSO.

## Tested

Run end to end on **n8n 2.45.4** (the version `docker-compose.yml` pins) and on
**n8n 1.123**, each with the GridWise API and a local SMTP server, using the
webhook-triggered variant (`--trigger webhook`) in place of IMAP:

- approve: the reviewer email arrived with working links, and the reply reached the customer
- reject: nothing was sent and the draft was recorded as rejected
- forged link (resume URL without the token): GridWise answered 403, nothing was sent,
  the draft stayed pending
- the customer's phone number appeared in neither the trace log nor the draft database

n8n 2.x signs its resume URLs (`?signature=...`); the workflow appends the decision
parameters with `&` in that case, so the same file works on both lines.
