"""Generate integrations/n8n/email_approval_workflow.json.

The workflow is generated from Python rather than hand-edited so that the
node wiring is readable in one place and stays consistent.

    python scripts/build_n8n_workflow.py [--gridwise-url http://gridwise:8000] [--trigger imap|webhook]

`--trigger webhook` swaps the IMAP trigger for an HTTP webhook; it is used by the
end-to-end test and is handy for local demos without a mailbox.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EMAIL = "New support email"
DRAFT = "Draft reply with GridWise"
ASK = "Ask reviewer to approve"
WAIT = "Wait for decision"
RECORD = "Record decision in GridWise"
IF_APPROVED = "Approved?"
SEND = "Send approved reply"
STOP = "Rejected or expired: send nothing"

# n8n 1.x resume URLs have no query string; n8n 2.x signs them (?signature=...),
# so the decision parameters are appended with '&' when a '?' is already present.
REVIEWER_TEXT = """=A customer email needs a reply. GridWise drafted one; nothing is sent until you decide.

Draft id:    {{ $json.draft_id }}
Confidence:  {{ $json.confidence }}
Flags:       {{ $json.flags.join(', ') || 'none' }}{{ $json.requires_attention ? '   <-- NEEDS ATTENTION' : '' }}
Sources:     {{ $json.citations.join(', ') || 'none' }}

Question (personal data redacted):
{{ $json.question }}

Draft reply:
----------------------------------------------------------------
{{ $json.draft_reply }}
----------------------------------------------------------------

Approve and send:  {{ $execution.resumeUrl }}{{ $execution.resumeUrl.includes('?') ? '&' : '?' }}decision=approve&reviewer=duty-energy-manager&token={{ $json.approval_token }}
Reject:            {{ $execution.resumeUrl }}{{ $execution.resumeUrl.includes('?') ? '&' : '?' }}decision=reject&reviewer=duty-energy-manager&token={{ $json.approval_token }}

These links are personal: anyone holding one can decide on this draft.

If nobody decides within 48 hours the draft expires and nothing is sent.
Trace of how this draft was produced: {{ $('""" + DRAFT + """').item.json.run_id }}
"""


def node(name: str, type_: str, version: float, position: list[int], parameters: dict, **extra) -> dict:
    return {"parameters": parameters, "name": name, "type": type_, "typeVersion": version,
            "position": position, "id": name.lower().replace(" ", "-").replace("?", "").replace(":", "")[:36], **extra}


def build(gridwise_url: str, trigger: str) -> dict:
    if trigger == "imap":
        source = node(EMAIL, "n8n-nodes-base.emailReadImap", 2.1, [0, 300], {
            "mailbox": "INBOX", "postProcessAction": "read", "format": "resolved", "options": {},
        }, credentials={"imap": {"id": "REPLACE_ME", "name": "Support mailbox (IMAP)"}})
        sender = f"$('{EMAIL}').item.json.from.value[0].address"
        subject = f"$('{EMAIL}').item.json.subject"
        draft_body = "={{ JSON.stringify({ subject: $json.subject || '', body: $json.text || '', message_id: $json.messageId || null }) }}"
    else:
        source = node(EMAIL, "n8n-nodes-base.webhook", 2, [0, 300], {
            "httpMethod": "POST", "path": "gridwise-email", "responseMode": "onReceived", "options": {},
        }, webhookId="gridwise-email-demo")
        sender = f"$('{EMAIL}').item.json.body.from"
        subject = f"$('{EMAIL}').item.json.body.subject"
        draft_body = "={{ JSON.stringify({ subject: $json.body.subject || '', body: $json.body.text || '', message_id: $json.body.messageId || null }) }}"

    smtp = {"smtp": {"id": "REPLACE_ME_SMTP", "name": "Outgoing mail (SMTP)"}}
    nodes = [
        source,
        node(DRAFT, "n8n-nodes-base.httpRequest", 4.2, [240, 300], {
            "method": "POST", "url": f"{gridwise_url}/email/draft",
            "sendBody": True, "specifyBody": "json", "jsonBody": draft_body, "options": {"timeout": 120000},
        }),
        node(ASK, "n8n-nodes-base.emailSend", 2.1, [480, 300], {
            "fromEmail": "gridwise@example.com",
            "toEmail": "duty-energy-manager@example.com",
            "subject": "=[GridWise draft {{ $json.draft_id }}]{{ $json.requires_attention ? ' NEEDS ATTENTION:' : '' }} {{ $json.subject }}",
            "emailFormat": "text", "text": REVIEWER_TEXT, "options": {"appendAttribution": False},
        }, credentials=smtp),
        node(WAIT, "n8n-nodes-base.wait", 1.1, [720, 300], {
            "resume": "webhook", "httpMethod": "GET", "responseMode": "onReceived",
            "limitWaitTime": True, "limitType": "afterTimeInterval", "resumeAmount": 48, "resumeUnit": "hours",
            "options": {},
        }, webhookId="gridwise-approval"),
        node(RECORD, "n8n-nodes-base.httpRequest", 4.2, [960, 300], {
            "method": "POST",
            "url": f"={gridwise_url}/drafts/{{{{ $('{DRAFT}').item.json.draft_id }}}}/decision",
            "sendBody": True, "specifyBody": "json",
            # A click on a link resumes the Wait node with the request ({headers, query, ...});
            # a timeout passes the previous item through, which has no headers. Only the
            # timeout path may use the draft's own token, and only to reject (expire).
            "jsonBody": (
                "={{ (() => { const clicked = $json.headers !== undefined; const q = $json.query || {}; "
                f"const draft = $('{DRAFT}').item.json; "
                "return JSON.stringify(clicked "
                "? { approved: q.decision === 'approve', reviewer: q.reviewer || 'unknown', approval_token: q.token || '' } "
                ": { approved: false, reviewer: 'expired-after-48h', approval_token: draft.approval_token, "
                "note: 'no decision within 48 hours' }); })() }}"
            ),
            "options": {},
        }),
        node(IF_APPROVED, "n8n-nodes-base.if", 2.2, [1200, 300], {
            "conditions": {
                "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
                "conditions": [{
                    "id": "approved-check", "leftValue": "={{ $json.status }}", "rightValue": "approved",
                    "operator": {"type": "string", "operation": "equals"},
                }],
                "combinator": "and",
            },
            "options": {},
        }),
        node(SEND, "n8n-nodes-base.emailSend", 2.1, [1440, 200], {
            "fromEmail": "support@example.com",
            "toEmail": f"={{{{ {sender} }}}}",
            "subject": f"=Re: {{{{ {subject} }}}}",
            "emailFormat": "text", "text": "={{ $json.final_reply }}", "options": {"appendAttribution": False},
        }, credentials=smtp),
        node(STOP, "n8n-nodes-base.noOp", 1, [1440, 420], {}),
    ]

    def link(target: str) -> list[dict]:
        return [{"node": target, "type": "main", "index": 0}]

    connections = {
        EMAIL: {"main": [link(DRAFT)]},
        DRAFT: {"main": [link(ASK)]},
        ASK: {"main": [link(WAIT)]},
        WAIT: {"main": [link(RECORD)]},
        RECORD: {"main": [link(IF_APPROVED)]},
        IF_APPROVED: {"main": [link(SEND), link(STOP)]},
    }
    return {
        "name": "GridWise: email -> draft -> human approval -> reply",
        "nodes": nodes,
        "connections": connections,
        "settings": {"executionOrder": "v1"},
        "pinData": {},
        "active": False,
        "meta": {"templateCredsSetupCompleted": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridwise-url", default="http://gridwise:8000")
    parser.add_argument("--trigger", choices=["imap", "webhook"], default="imap")
    parser.add_argument("--out", type=Path, default=ROOT / "integrations" / "n8n" / "email_approval_workflow.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build(args.gridwise_url, args.trigger), indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
