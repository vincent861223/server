"""Unit tests for the Apple Music setup flow helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError

from music_assistant.providers.apple_music import setup_flow
from music_assistant.providers.apple_music.constants import (
    CONF_MUSIC_APP_TOKEN,
    CONF_MUSIC_USER_TOKEN,
)
from music_assistant.providers.apple_music.setup_flow import _app_token_accepted, run_setup


class _FakeRequestCtx:
    """Minimal async context manager mimicking aiohttp's request context."""

    def __init__(self, status: int) -> None:
        self._response = MagicMock(status=status)

    async def __aenter__(self) -> MagicMock:
        return self._response

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _make_mass(status: int) -> MagicMock:
    mass = MagicMock()
    mass.http_session.get = MagicMock(return_value=_FakeRequestCtx(status))
    return mass


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, True),
        (401, False),
        (403, False),
        # a throttled or broken /v1/test says nothing about the token itself
        (429, None),
        (500, None),
        (503, None),
    ],
)
@pytest.mark.asyncio
async def test_app_token_accepted_per_status(status: int, expected: bool | None) -> None:
    """Only an explicit rejection by Apple marks the developer token as invalid."""
    assert await _app_token_accepted(_make_mass(status), "app-token") is expected


@pytest.mark.asyncio
async def test_app_token_accepted_empty_token_is_rejected() -> None:
    """An empty (e.g. not provisioned) token is rejected without calling the API."""
    mass = _make_mass(200)
    assert await _app_token_accepted(mass, "") is False
    mass.http_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_app_token_accepted_network_error_is_inconclusive() -> None:
    """An unreachable API does not make a valid token look invalid."""
    mass = MagicMock()
    mass.http_session.get = MagicMock(side_effect=ClientError("boom"))
    assert await _app_token_accepted(mass, "app-token") is None


def _make_session(
    form_values: list[dict[str, str]], setup_data: dict[str, str] | None = None
) -> MagicMock:
    session = MagicMock()
    session.context.setup_data = setup_data or {}
    session.form = AsyncMock(side_effect=form_values)
    session.finish = AsyncMock()
    return session


async def _run_setup(session: MagicMock, accepted: dict[str, bool | None]) -> dict[str, object]:
    """Run the flow with per-token validation results and return the values it finished with."""

    async def _accepted(_mass: object, app_token: str) -> bool | None:
        return accepted[app_token]

    with (
        patch.object(setup_flow, "_app_token_accepted", _accepted),
        patch.object(setup_flow, "MUSIC_APP_TOKEN", "bundled-token"),
        patch.object(
            setup_flow,
            "_musickit_authenticate",
            AsyncMock(return_value={"music-user-token": "user-token"}),
        ),
    ):
        await run_setup(session)
    return dict(session.finish.call_args.args[0])


@pytest.mark.asyncio
async def test_run_setup_clears_a_stored_app_token_that_apple_rejects() -> None:
    """Signing in again drops a stored app token that expired, so the bundled one is used."""
    session = _make_session([{}], setup_data={CONF_MUSIC_APP_TOKEN: "stored-token"})
    values = await _run_setup(session, {"stored-token": False, "bundled-token": True})
    # the reconfigure finish handler merges into the old setup_data, so the key must be
    # present (and empty) to remove the stored token
    assert CONF_MUSIC_APP_TOKEN in values
    assert values[CONF_MUSIC_APP_TOKEN] is None
    assert values[CONF_MUSIC_USER_TOKEN] == "user-token"


@pytest.mark.parametrize("stored_valid", [True, None])
@pytest.mark.asyncio
async def test_run_setup_keeps_a_stored_app_token_that_is_not_rejected(
    stored_valid: bool | None,
) -> None:
    """A working (or not verifiable) custom developer token survives signing in again."""
    session = _make_session([{}], setup_data={CONF_MUSIC_APP_TOKEN: "stored-token"})
    values = await _run_setup(session, {"stored-token": stored_valid, "bundled-token": True})
    assert CONF_MUSIC_APP_TOKEN not in values


@pytest.mark.asyncio
async def test_run_setup_stores_manual_app_token_when_bundled_is_rejected() -> None:
    """A typed app token replaces a rejected stored one when the bundled one is rejected too."""
    session = _make_session(
        [{CONF_MUSIC_APP_TOKEN: "typed-token"}, {}],
        setup_data={CONF_MUSIC_APP_TOKEN: "stored-token"},
    )
    values = await _run_setup(
        session, {"stored-token": False, "bundled-token": False, "typed-token": True}
    )
    assert values[CONF_MUSIC_APP_TOKEN] == "typed-token"
