"""Local, deliberately small channel setup store.

This is an installer UI aid, not a cloud credential vault.  Secrets are kept
only on the gateway machine in ``$GLC_CONFIG_DIR/channel_secrets.json`` with
owner-only permissions.  Read APIs expose *presence*, never values.  At boot
the saved values are made available to the existing adapters through their
documented environment variable names; an explicit process environment value
always wins.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from glc import config

logger = logging.getLogger("glc.channels.setup")

CHANNEL_SPECS: dict[str, dict[str, Any]] = {
    # Guide-only cards intentionally have no saveable fields: these adapters
    # currently consume an injected transport/client, not env credentials.
    "discord": {"mode": "external Discord gateway bridge", "docs": "https://discord.com/developers/applications", "fields": [], "steps": ["Create an application and bot in the Discord developer portal.", "Invite it to the target server, then run a Discord gateway bridge that injects the adapter's `client` config. GLC does not consume DISCORD_BOT_TOKEN itself."], "webhook": False},
    "gmail": {"mode": "OAuth helper", "docs": "https://console.cloud.google.com/apis/library/gmail.googleapis.com", "fields": [("GMAIL_OAUTH_CLIENT_ID", "OAuth client ID", True), ("GMAIL_OAUTH_CLIENT_SECRET", "OAuth client secret", True), ("GMAIL_BOT_ADDRESS", "Bot Gmail address", False)], "steps": ["Enable Gmail API and Pub/Sub in Google Cloud.", "Save the downloaded desktop OAuth credentials as gmail/credentials.json and run `python -m glc.channels.catalogue.gmail.auth_setup` locally.", "A saved client ID/secret is consumed by the adapter, but the OAuth token and Gmail watch remain a manual provider step."], "webhook": False},
    "imap": {"mode": "external IMAP/SMTP transport", "docs": "https://www.rfc-editor.org/rfc/rfc9051", "fields": [], "steps": ["Use an app password, not your normal mailbox password.", "Run a polling/IMAP service that injects the adapter's `smtp_host`, `smtp_user`, and `smtp_password` config. GLC does not consume IMAP/SMTP environment fields for this adapter."], "webhook": False},
    "line": {"mode": "external LINE transport", "docs": "https://developers.line.biz/console/", "fields": [], "steps": ["Create a Messaging API channel in LINE Developers.", "Run a LINE bridge that validates signatures and injects the adapter's `transport` config. GLC does not consume LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET itself."], "webhook": False},
    "local_mic": {"mode": "external local microphone client", "docs": None, "fields": [], "steps": ["No external authentication is required.", "Run a local microphone capture client that sends audio frames to GLC and configure STT/TTS preferences in that client's adapter config. GLC does not consume LOCAL_MIC_OWNER_ID."], "webhook": False},
    "matrix": {"mode": "external Matrix sync client", "docs": "https://matrix.org/docs/communities/administration/", "fields": [], "steps": ["Create a Matrix bot account and access token.", "Run a Matrix sync client and inject it as the adapter transport; this adapter has no built-in homeserver/token client."], "webhook": False},
    "signal": {"mode": "external signal-cli bridge", "docs": "https://github.com/AsamK/signal-cli", "fields": [], "steps": ["Register a dedicated number with signal-cli.", "Run signal-cli JSON-RPC separately and bridge events to the adapter; this adapter does not consume Signal credentials itself."], "webhook": False},
    "slack": {"mode": "external Slack Events bridge", "docs": "https://api.slack.com/apps", "fields": [], "steps": ["Create a Slack app, add bot scopes, and install it to the workspace.", "Run an Events API/Socket Mode bridge that verifies Slack signatures and passes event dictionaries to the adapter. This adapter currently has no native token client."], "webhook": False},
    "teams": {"mode": "Bot Framework outbound credentials", "docs": "https://learn.microsoft.com/azure/bot-service/", "fields": [("TEAMS_APP_ID", "Microsoft app ID", True), ("TEAMS_APP_PASSWORD", "Microsoft app password", True), ("TEAMS_TENANT_ID", "Microsoft tenant ID", True)], "steps": ["Register a Bot Framework application and Teams channel.", "These exact values are consumed for outbound Connector OAuth. A Bot Framework receiver must still validate inbound activities and pass their JSON to the adapter; GLC's generic webhook does not do that validation."], "webhook": False},
    "telegram": {"mode": "Telegram Bot API credential", "docs": "https://core.telegram.org/bots", "fields": [("TELEGRAM_BOT_TOKEN", "Bot token", True)], "steps": ["Create a bot with @BotFather.", "GLC consumes the token for Bot API file/download and outbound send. Run a polling bridge or verified update receiver for inbound Telegram updates; the generic webhook route does not parse Telegram updates."], "webhook": False},
    "twilio_sms": {"mode": "Twilio SMS credential", "docs": "https://console.twilio.com/", "fields": [("TWILIO_ACCOUNT_SID", "Account SID", True), ("TWILIO_AUTH_TOKEN", "Auth token", True), ("TWILIO_PHONE_NUMBER", "Twilio SMS number", False)], "steps": ["Buy/configure a Twilio number with SMS capability.", "GLC consumes these values for outbound SMS and the adapter reads Twilio form dictionaries. Run a Twilio receiver that validates/parses the form before calling the adapter; the generic webhook route passes raw bytes."], "webhook": False},
    "twilio_voice": {"mode": "Twilio Voice signature credential", "docs": "https://console.twilio.com/", "fields": [("TWILIO_AUTH_TOKEN", "Auth token", True)], "steps": ["Configure a Twilio Voice number and Media Streams separately.", "GLC consumes this token to verify a call event. The adapter builds TwiML replies but does not place calls; a Twilio receiver/stream service must parse and deliver the events."], "webhook": False},
    "webhook": {"mode": "GLC native signed HTTP webhook", "docs": None, "fields": [("WEBHOOK_SHARED_SECRET", "Inbound shared secret", True), ("WEBHOOK_DEFAULT_TARGET_URL", "Outbound target URL", False)], "steps": ["Send JSON matching the WebhookInbound schema with the adapter's timestamped X-Webhook-Signature.", "GLC directly verifies this format at the generated ingress URL. Configure a public HTTPS base URL before using it externally."], "webhook": True},
    "webui": {"mode": "built-in WebSocket", "docs": None, "fields": [], "steps": ["No provider credential is required.", "Connect the application UI to GLC's authenticated WebSocket channel endpoint."], "webhook": False},
    "whatsapp": {"mode": "Meta Cloud API + native webhook", "docs": "https://developers.facebook.com/apps/", "fields": [("WHATSAPP_TOKEN", "Cloud API access token", True), ("WHATSAPP_APP_SECRET", "App secret", True), ("WHATSAPP_PHONE_NUMBER_ID", "Phone number ID", False), ("WHATSAPP_VERIFY_TOKEN", "Meta webhook verify token", True)], "steps": ["Create a Meta app and WhatsApp Business Cloud API phone number.", "GLC validates Meta POST signatures using WHATSAPP_APP_SECRET and consumes the Cloud API token for outbound delivery. Use the callback URL below and its matching GET verification token."], "webhook": True, "verification": "meta_hub"},
}


def _path() -> Path:
    return config.CONFIG_DIR / "channel_secrets.json"


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"channels": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"channels": {}}
    return data if isinstance(data, dict) and isinstance(data.get("channels"), dict) else {"channels": {}}


def _restrict_to_owner(path: Path) -> None:
    """Make the secret file readable only by its owner, on every platform.

    POSIX chmod expresses this directly. Windows does not: os.chmod there only
    toggles the read-only attribute and silently ignores the rest, so it neither
    restricts the file nor raises. An `except OSError` around it therefore never
    fires on Windows and the file is left world-readable while the code believes
    it succeeded. icacls is used instead, and if that cannot be applied the
    caller is warned rather than left with a guarantee that does not hold.
    """
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    if os.name != "nt":
        return

    account = os.environ.get("USERNAME") or ""
    if not account:
        logger.warning("cannot restrict %s: USERNAME is not set", path)
        return
    try:
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:F"],
            capture_output=True, text=True, timeout=15, check=False)
        if result.returncode != 0:
            logger.warning("could not restrict %s to %s: %s", path, account,
                           (result.stderr or result.stdout or "").strip()[:200])
    except Exception as error:  # pragma: no cover - depends on the host
        logger.warning("could not restrict %s to %s: %s: %s",
                       path, account, type(error).__name__, error)


def _save(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.chmod(temp, 0o600)
    temp.replace(path)
    _restrict_to_owner(path)


def configured(name: str) -> dict[str, Any]:
    return dict(_load()["channels"].get(name, {}))


def update(name: str, values: dict[str, str], enabled: bool | None = None) -> None:
    spec = CHANNEL_SPECS[name]
    allowed = {field[0] for field in spec["fields"]}
    data = _load()
    record = dict(data["channels"].get(name, {}))
    secrets = dict(record.get("values", {}))
    for key, value in values.items():
        if key in allowed and isinstance(value, str):
            if value:
                secrets[key] = value
            elif key in secrets:
                del secrets[key]
    record["values"] = secrets
    if enabled is not None:
        record["enabled"] = enabled
    data["channels"][name] = record
    _save(data)


def public_catalogue(base_url: str, known_channels: list[str]) -> list[dict[str, Any]]:
    saved = _load()["channels"]
    result: list[dict[str, Any]] = []
    for name in known_channels:
        spec = CHANNEL_SPECS.get(name, {"mode": "adapter configuration", "fields": [], "steps": [], "webhook": False})
        record = saved.get(name, {})
        values = record.get("values", {})
        fields = [
            {"key": key, "label": label, "secret": secret, "configured": bool(values.get(key)) or bool(os.getenv(key))}
            for key, label, secret in spec["fields"]
        ]
        can_configure = bool(spec["fields"])
        configured = can_configure and bool(record.get("enabled", False)) and all(
            field["configured"] for field in fields
        )
        result.append({"name": name, "enabled": bool(record.get("enabled", False)), "can_configure": can_configure, "configured": configured, "mode": spec["mode"], "docs": spec.get("docs"), "fields": fields, "steps": spec["steps"], "webhook_url": f"{base_url.rstrip('/')}/v1/channels/{name}/webhook" if spec["webhook"] else None, "verification": spec.get("verification")})
    return result


def apply_saved_environment() -> None:
    """Expose stored credentials to legacy adapters without exposing them over HTTP."""
    for name, record in _load()["channels"].items():
        if not record.get("enabled", False):
            continue
        spec = CHANNEL_SPECS.get(name)
        if spec is None:
            continue
        allowed = {field[0] for field in spec["fields"]}
        for key, value in record.get("values", {}).items():
            if key in allowed and isinstance(value, str) and value:
                os.environ.setdefault(key, value)
