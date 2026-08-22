"""Gemini Live (streaming voice in via WebSocket) STT provider.

Google's BidiGenerateContent endpoint is full-duplex: the client opens a
WebSocket, sends a `BidiGenerateContentSetup` frame *first*, then streams
audio, and reads transcript chunks back until the server emits
`turnComplete`.

This module has two paths:

* ``_transcribe_via_mock`` — used by the CI test-suite. When
  ``config["mock"]`` is present the adapter talks to the in-repo fake in
  ``tests/voice/stt/mocks/gemini_live_mock.py`` instead of the network.
  This is the path the 7 tests exercise.

* ``_transcribe_via_websocket`` — the real upstream call used outside the
  tests (e.g. for the demo video). CI never runs it.

Both paths obey the same wire rule: the ``setup`` frame is always sent
before the audio frame.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import time
from typing import Any

from glc.voice.stt.base import STTError, STTProvider, TranscribeResult
from glc.voice.stt.providers.gemini_live.schemas import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_MODALITIES,
    PCM_MIME_TYPE,
    WAV_HEADER_BYTES,
    WS_ENDPOINT,
    GeminiLiveAudioFrame,
    GeminiLiveAudioInput,
    GeminiLiveAudioStreamEndFrame,
    GeminiLiveGenerationConfig,
    GeminiLiveRealtimeInput,
    GeminiLiveSetupFrame,
    GeminiLiveSetupPayload,
    GeminiLiveSystemInstruction,
    GeminiLiveTextPart,
)

# Upper bound on a single inbound WebSocket frame. The TTS sibling adapter
# keeps the websockets default; this one had opted out of any cap at all.
_WS_MAX_FRAME_BYTES = 16 * 1024 * 1024


class Provider(STTProvider):
    name = "gemini_live"

    # ── public entry point ─────────────────────────────────────────
    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        """Turn an audio clip into text.

        Picks the fake transport when a mock is injected (tests), and the
        real WebSocket transport otherwise.
        """
        mock = self.config.get("mock")
        if mock is not None:
            return await self._transcribe_via_mock(mock, audio, mime)
        return await self._transcribe_via_websocket(audio, mime)

    # ── frame builders (shared by both paths) ──────────────────────
    def _build_setup_frame(self) -> dict[str, Any]:
        """The BidiGenerateContentSetup frame. Must be sent first.

        ``responseModalities`` lives under ``generationConfig`` in the real
        v1beta wire format; the canonical key the tests look for is the
        top-level ``setup`` key, which is preserved.
        """
        model = self.config.get("model", DEFAULT_MODEL)
        # gemini-3.1-flash-live-preview only supports AUDIO responseModalities.
        # TEXT modality is rejected by this model (1007 error).
        # inputAudioTranscription: {} is also rejected by the current API (1007).
        # Strategy: use outputAudioTranscription to get a text transcript of the
        # model's AUDIO reply. A systemInstruction tells the model to repeat
        # the user's words verbatim, so the output transcript == the input STT.
        modalities = self.config.get("response_modalities", DEFAULT_RESPONSE_MODALITIES)
        payload = GeminiLiveSetupPayload(
            model=model,
            generationConfig=GeminiLiveGenerationConfig(responseModalities=modalities),
            outputAudioTranscription={},
            systemInstruction=GeminiLiveSystemInstruction(
                parts=[
                    GeminiLiveTextPart(
                        text=(
                            "You are a speech transcription service. "
                            "Repeat back exactly what the user says, "
                            "word for word. Output only the transcription "
                            "with no additional commentary."
                        )
                    )
                ]
            ),
        )
        return GeminiLiveSetupFrame(setup=payload).model_dump(by_alias=True)

    def _build_audio_frame(self, audio: bytes, mime: str) -> dict[str, Any]:
        """The realtimeInput frame carrying the (base64) audio payload.

        Gemini Live requires raw PCM (not a WAV container). If the caller
        passes WAV bytes (detected by the 'RIFF' magic header), the 44-byte
        header is stripped automatically before encoding.
        """
        # Strip WAV container header if present — Gemini Live expects raw PCM.
        # WAV files start with the 4-byte ASCII magic 'RIFF'.
        if audio[:4] == b"RIFF":
            audio = audio[WAV_HEADER_BYTES:]
            mime = PCM_MIME_TYPE
        encoded = base64.b64encode(audio).decode("ascii")
        payload = GeminiLiveAudioFrame(
            realtimeInput=GeminiLiveRealtimeInput(audio=GeminiLiveAudioInput(mimeType=mime, data=encoded))
        )
        return payload.model_dump(by_alias=True)

    # ── mock path (what CI exercises) ──────────────────────────────
    async def _transcribe_via_mock(self, mock: Any, audio: bytes, mime: str) -> TranscribeResult:
        """Drive the in-repo fake upstream.

        Order matters: the setup frame is recorded before the audio frame
        so ``frames_sent[0]`` is always the setup frame (the Live API
        rejects sessions where audio arrives first).
        """
        mock.record_frame(self._build_setup_frame())
        mock.record_frame(self._build_audio_frame(audio, mime))
        # The fake returns a canned TranscribeResult or raises STTError;
        # let the error propagate untouched.
        return await mock.transcribe(audio, mime)

    # ── real path (for the demo; not run by CI) ────────────────────
    async def _transcribe_via_websocket(self, audio: bytes, mime: str) -> TranscribeResult:
        """Real Gemini Live call over the BidiGenerateContent WebSocket.

        Flow (per https://ai.google.dev/api/multimodal-live):

          1. Open ``{_WS_ENDPOINT}?key=$GEMINI_API_KEY``.
          2. Send the setup frame FIRST (includes outputAudioTranscription
             and systemInstruction to transcribe the reply verbatim).
          3. Send the audio as a ``realtimeInput.audio`` frame with raw PCM
             at 16 kHz (WAV header stripped if present), then signal
             ``audioStreamEnd`` so the server closes the input turn.
          4. Read messages, accumulating text from
             ``serverContent.outputTranscription.text`` (primary), falling
             back to ``inputTranscription`` or ``modelTurn.parts[].text``.
             Ignore ``setupComplete`` / ``usageMetadata`` frames. Stop on
             ``serverContent.turnComplete``.
          5. Wrap any failure in ``STTError``.

        Requires ``GEMINI_API_KEY`` in the environment (or ``config``).
        """
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - dependency present
            raise STTError("the 'websockets' package is required", status=None) from exc

        api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1")
        if not api_key:
            raise STTError("GEMINI_API_KEY is not set", status=None)

        url = f"{WS_ENDPOINT}?key={api_key}"
        start = time.monotonic()
        transcript: list[str] = []

        # Allow disabling SSL verification for corporate proxies that inject
        # non-standard CA certificates (set config["ssl_verify"] = False).
        ssl_ctx: ssl.SSLContext | bool = True
        if self.config.get("ssl_verify") is False:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        try:
            # max_size=None disables the library's frame cap entirely, so a
            # misbehaving upstream (or anything intercepting the connection)
            # can hand us a frame large enough to exhaust memory. The
            # transcript frames this reads are small; 16 MiB is far above any
            # legitimate one and still bounded.
            async with websockets.connect(url, max_size=_WS_MAX_FRAME_BYTES, ssl=ssl_ctx) as ws:
                # 1. setup must be the first frame
                await ws.send(json.dumps(self._build_setup_frame()))
                # 2. push the audio, then close the input turn
                await ws.send(json.dumps(self._build_audio_frame(audio, mime)))
                await ws.send(
                    json.dumps(
                        GeminiLiveAudioStreamEndFrame(
                            realtimeInput=GeminiLiveRealtimeInput(audioStreamEnd=True)
                        ).model_dump(by_alias=True)
                    )
                )
                # 3. drain responses until the turn completes.
                # Lock onto the first field that produces text and ignore all
                # others for the rest of the session — prevents the same
                # sentence appearing twice when both outputTranscription and
                # modelTurn fire in separate messages.
                preferred_source: str | None = None
                async for raw in ws:
                    data = json.loads(raw)
                    server_content = data.get("serverContent")
                    if not server_content:
                        # Skip non-content frames: setupComplete, usageMetadata,
                        # sessionResumptionUpdate, etc.
                        continue
                    output_tx = server_content.get("outputTranscription")
                    input_tx = server_content.get("inputTranscription")
                    model_turn = server_content.get("modelTurn")

                    if output_tx and output_tx.get("text"):
                        preferred_source = preferred_source or "outputTranscription"
                        if preferred_source == "outputTranscription":
                            transcript.append(output_tx["text"])
                    elif input_tx and input_tx.get("text"):
                        preferred_source = preferred_source or "inputTranscription"
                        if preferred_source == "inputTranscription":
                            transcript.append(input_tx["text"])
                    elif model_turn and preferred_source is None:
                        for part in model_turn.get("parts", []):
                            text = part.get("text")
                            if text:
                                transcript.append(text)
                        if transcript:
                            preferred_source = "modelTurn"

                    if server_content.get("turnComplete"):
                        break
        except STTError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any upstream failure
            raise STTError(f"Gemini Live WebSocket error: {exc}", status=None) from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        return TranscribeResult(
            text="".join(transcript),
            language=self.config.get("language", DEFAULT_LANGUAGE),
            duration_ms=duration_ms,
            provider=self.name,
            cost_usd=0.0,
        )
